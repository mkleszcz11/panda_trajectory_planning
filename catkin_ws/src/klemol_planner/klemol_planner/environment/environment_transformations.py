import numpy as np
import math
from scipy.spatial.transform import Rotation as R
from klemol_planner.goals.point_with_orientation import PointWithOrientation
from klemol_planner.camera_utils.camera_operations import CameraOperations

import tf.transformations as tf_trans
import math

import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches

class PandaTransformations:
    def __init__(self, cam_operations):
        # Table corners in base frame
        # Found with command:
        # rosrun tf tf_echo /panda_link0 /panda_link8
        self.camera_operations = cam_operations

        # This is the height from flange (last coordinate system) to the table while we are calibrating
        self.z_calibration_constant = 0.12

        self.table_corners_translations = {
            "corner_0": np.array([0.687, -0.385, 0.177 - self.z_calibration_constant]),
            "corner_1": np.array([0.674,  0.366, 0.183 - self.z_calibration_constant]),
            "corner_2": np.array([0.100,  0.412, 0.195 - self.z_calibration_constant]),
            "corner_3": np.array([0.099, -0.412, 0.196 - self.z_calibration_constant])
        }
        # Table corners, rpy in radians:
        # We are not using quaternions as it is easy to rotate only in z axis for rpy
        self.table_corners_rpy = {
            "corner_0": [0.0, 0.0, -0.788 + math.pi * 0.75],
            "corner_1": [0.0, 0.0, -0.788 + math.pi * 0.75],
            "corner_2": [0.0, 0.0, -0.791 + math.pi * 0.75],
            "corner_3": [0.0, 0.0, -0.761 + math.pi * 0.75]
        }


        # Compute transformation matrices for each corner
        self.T_base_to_corners_dict = {}
        for corner_name, translation in self.table_corners_translations.items():
            rpy = self.table_corners_rpy[corner_name]
            rotation_matrix = tf_trans.euler_matrix(*rpy)  # returns 4x4
            transform = np.eye(4)
            transform[:3, :3] = rotation_matrix[:3, :3]  # copy rotation
            transform[:3, 3] = translation  # set translation
            self.T_base_to_corners_dict[corner_name] = transform

        # Print all possible base to camera dict #
        for btc in self.T_base_to_corners_dict:
            formatted = np.array2string(
                self.T_base_to_corners_dict[btc],
                formatter={'float_kind': lambda x: f"{x:.2f}"}
            )
            print(f"base to corner {btc} =\n{formatted}")

        # self.T_table_corners_to_camera_dict ={
        #     "corner_0": np.ones(4),
        #     "corner_1": np.ones(4),
        #     "corner_2": np.ones(4),
        #     "corner_3": np.ones(4)}

        # self.calibrate_camera_translation()
        # self.calibrate_camera_rotation()

        # print(f"-----------------------------------------")
        # # Print all corner to camera dict 
        # for btc in self.T_table_corners_to_camera_dict:
        #     formatted = np.array2string(
        #         self.T_table_corners_to_camera_dict[btc],
        #         formatter={'float_kind': lambda x: f"{x:.2f}"}
        #     )
        #     print(f"corner {btc} to camera =\n{formatted}")
        # print(f"-----------------------------------------")


        # # T_base_to_camera_dict = T_base_to_corners_dict * T_table_corners_to_camera_dict
        # # T_table_corners_to_camera_dict = np.linalg.inv(self.T_camera_to_table)
        # self.T_base_to_camera_dict = {
        #     "corner_0" : self.T_base_to_corners_dict["corner_0"] @ self.T_table_corners_to_camera_dict["corner_0"],
        #     "corner_1" : self.T_base_to_corners_dict["corner_1"] @ self.T_table_corners_to_camera_dict["corner_1"],
        #     "corner_2" : self.T_base_to_corners_dict["corner_2"] @ self.T_table_corners_to_camera_dict["corner_2"],
        #     "corner_3" : self.T_base_to_corners_dict["corner_3"] @ self.T_table_corners_to_camera_dict["corner_3"]}

        # self.T_base_to_camera = self.get_average_transform_to_camera()

        # Initialize storage for camera transformations per corner
        self.T_table_corners_to_camera_dict = {}
        self.T_base_to_camera_dict = {}

        self.calibrate_camera() # T_base_to_camera is computed here

        # Print all possible base to camera dict #
        for corner in self.T_base_to_camera_dict:
            formatted = np.array2string(
                self.T_base_to_camera_dict[corner],
                formatter={'float_kind': lambda x: f"{x:.2f}"}
            )
            print(f"transformation based on {corner} =\n{formatted}")

        # Transformation from camera to object frame will be updated based on object pose
        self.T_camera_to_object = np.eye(4)

    def calibrate_camera(self) -> None:
        """
        Estimate base to camera transformation using ArUco markers.
        Visualize the process step-by-step:
        1. Before centering
        2. After centering
        3. After rotation
        """
        print("[INFO] Calibrating camera...")
        camera_op = self.camera_operations
        detected_markers = camera_op.get_marker_transforms()

        print(f"[INFO] I GOT MARKERS: {detected_markers}")

        if len(detected_markers) < 4:
            raise RuntimeError("Not all 4 corners were detected")

        base_xy = []
        cam_xy = []
        cam_zs = []

        # # Variable to store camera yaw rotation (related to base frame).
        # camera_rotation = 0.0

        for corner_name, T_cam_marker in detected_markers.items():
            if corner_name not in self.table_corners_translations:
                continue
            p_cam = T_cam_marker[:3, 3]
            p_base = self.table_corners_translations[corner_name]

            cam_xy.append(p_cam[:2])
            base_xy.append(p_base[:2])
            cam_zs.append(p_cam[2] + p_base[2])

        # Z Axis for camera points the other way as for base
        # # MAKE A "MIRROR VIEW" along the Z axis for x and y
        cam_xy = np.array(cam_xy) * np.array([1, -1])

        base_xy = np.array(base_xy).T  # shape (2, N) -> np.array([[x1, x2, x3, x4], [y1, y2, y3, y4]])
        cam_xy = np.array(cam_xy).T

        base_mean = np.mean(base_xy, axis=1, keepdims=True)
        cam_mean = np.mean(cam_xy, axis=1, keepdims=True)

        base_centered = base_xy - base_mean
        cam_centered = cam_xy - cam_mean

        ###############################
        ####### START PLOTTING #######
        ###############################
        # PLOT CORNERS RELATED TO BASE & CAMERA BEFORE CENTERING
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle("Step 1: Before Centering (Separate Frames)")

        # BASE FRAME PLOT
        ax1.set_title("Base Frame")
        ax1.scatter(base_xy[0], base_xy[1], c='black', label='Base Corners')
        for i in range(base_xy.shape[1]):
            ax1.text(base_xy[0, i], base_xy[1, i], f"B{i}", color='black')

        # Draw base axes (origin at 0,0)
        ax1.quiver(0, 0, 0.1, 0, color='red', angles='xy', scale_units='xy', scale=1, label='Base X')
        ax1.quiver(0, 0, 0, 0.1, color='green', angles='xy', scale_units='xy', scale=1, label='Base Y')

        ax1.set_aspect('equal')
        ax1.grid(True)
        ax1.legend()
        ax1.set_xlabel("X [m]")
        ax1.set_ylabel("Y [m]")

        # CAMERA FRAME PLOT
        ax2.set_title("Camera Frame")
        ax2.scatter(cam_xy[0], cam_xy[1], c='orange', label='Camera Corners')
        for i in range(cam_xy.shape[1]):
            ax2.text(cam_xy[0, i], cam_xy[1, i], f"C{i}", color='orange')

        # Draw camera axes (origin at 0,0)
        ax2.quiver(0, 0, 0.1, 0, color='red', angles='xy', scale_units='xy', scale=1, label='Cam X')
        ax2.quiver(0, 0, 0, -0.1, color='green', angles='xy', scale_units='xy', scale=1, label='Cam Y')

        ax2.set_aspect('equal')
        ax2.grid(True)
        ax2.legend()
        ax2.set_xlabel("X [m]")
        ax2.set_ylabel("Y [m]")

        plt.tight_layout()
        # plt.show()
        ###############################
        ####### FINISH PLOTTING #######
        ###############################


        ##############################
        ####### START PLOTTING #######
        ##############################
        # PLOT TWO PREVIOUS PLOTS ON TOP OF EACH OTHER (BEFORE ROTATION)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title("Step 2: Base vs Camera Corners (Before Alignment)")

        # Plot base corners
        ax.scatter(base_xy[0], base_xy[1], c='black', label='Base Corners')
        for i in range(base_xy.shape[1]):
            ax.text(base_xy[0, i], base_xy[1, i], f"B{i}", color='black')

        # Plot camera corners (after Y-flip)
        ax.scatter(cam_xy[0], cam_xy[1], c='orange', label='Camera Corners')
        for i in range(cam_xy.shape[1]):
            ax.text(cam_xy[0, i], cam_xy[1, i], f"C{i}", color='orange')

        # Draw base frame at (0,0)
        ax.quiver(0, 0, 0.1, 0, color='red', angles='xy', scale_units='xy', scale=1, label='Base X')
        ax.quiver(0, 0, 0, 0.1, color='green', angles='xy', scale_units='xy', scale=1, label='Base Y')

        # Camera frame at camera origin (0,0)
        ax.quiver(0, 0, 0.1, 0, color='blue', linestyle='dashed', scale_units='xy', scale=1, label='Cam X')
        ax.quiver(0, 0, 0, -0.1, color='cyan', linestyle='dashed', scale_units='xy', scale=1, label='Cam Y')

        ax.set_aspect('equal')
        ax.grid(True)
        ax.legend()
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        plt.tight_layout()
        # plt.show()
        ###############################
        ####### FINISH PLOTTING #######
        ###############################

        # Apply rotation (Kabsch)
        H = cam_centered @ base_centered.T
        U, _, Vt = np.linalg.svd(H)
        R_2D = Vt.T @ U.T
        if np.linalg.det(R_2D) < 0:
            Vt[1, :] *= -1
            R_2D = Vt.T @ U.T

        ##############################
        ####### START PLOTTING #######
        ##############################
        # PLOT EVERYTHING AFTER ROTATION

        # Apply rotation to camera points
        cam_rotated = R_2D @ cam_centered

        # Re-shift camera points to base frame
        cam_aligned = cam_rotated + base_mean

        # Plot base vs rotated camera
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title("Step 3: After Kabsch Rotation (Aligned)")

        # Base
        ax.scatter(base_xy[0], base_xy[1], c='black', label='Base Corners')
        for i in range(base_xy.shape[1]):
            ax.text(base_xy[0, i], base_xy[1, i], f"B{i}", color='black')

        # Rotated & aligned camera corners
        ax.scatter(cam_aligned[0], cam_aligned[1], c='orange', label='Rotated Camera Corners')
        for i in range(cam_aligned.shape[1]):
            ax.text(cam_aligned[0, i], cam_aligned[1, i], f"C{i}", color='orange')

        # Base frame (origin at 0,0)
        ax.quiver(0, 0, 0.1, 0, color='red', angles='xy', scale_units='xy', scale=1, label='Base X')
        ax.quiver(0, 0, 0, 0.1, color='green', angles='xy', scale_units='xy', scale=1, label='Base Y')

        # Camera frame (estimated position = base_mean, estimated orientation = R_2D)
        cam_frame_origin = base_mean.flatten()
        cam_axes = R_2D @ np.eye(2) * 0.1
        ax.quiver(*cam_frame_origin, cam_axes[0, 0], cam_axes[1, 0], color='blue', label='Cam X')
        ax.quiver(*cam_frame_origin, cam_axes[0, 1], cam_axes[1, 1], color='cyan', label='Cam Y')

        ax.set_aspect('equal')
        ax.grid(True)
        ax.legend()
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        plt.tight_layout()
        # plt.show()

        # Final transformation
        yaw = np.arctan2(R_2D[1, 0], R_2D[0, 0])
        z_camera = np.mean(cam_zs)

        T_base_to_camera_before_flip = np.array([
            [np.cos(yaw), -np.sin(yaw), 0, cam_frame_origin[0]],
            [np.sin(yaw),  np.cos(yaw), 0, cam_frame_origin[1]],
            [0,           0,            1, z_camera],
            [0,           0,            0, 1]
        ])
        X_flip_matrix = np.array([
            [1,  0,  0, 0],
            [0, -1,  0, 0],
            [0,  0, -1, 0],
            [0,  0,  0, 1]
        ])

        self.T_base_to_camera = T_base_to_camera_before_flip @ X_flip_matrix

        print(f"[INFO] T_base_to_camera estimated from corners:\n{np.round(self.T_base_to_camera, 4)}")



    # def calibrate_camera_v2(self) -> None:
    #     """
    #     Calibrate camera using ArUco markers (each at a table corner).
    #     """
    #     camera_op = CameraOperations()
    #     try:
    #         markers = camera_op.find_aruco_codes_in_the_image()
    #     except Exception:
    #         raise RuntimeError("CAMERA NOT CALIBRATED")

    #     self.T_table_corners_to_camera_dict = {}

    #     for marker_id, tvec, rvec in markers:
    #         yaw = rvec[2]  # Only Z rotation

    #         R_z = np.array([
    #             [np.cos(yaw), -np.sin(yaw), 0],
    #             [np.sin(yaw),  np.cos(yaw), 0],
    #             [0,            0,           1]
    #         ])

    #         # Flip 180° about X to align Z down, then rotate 270° clockwise around Z
    #         R_flip = np.array([[1, 0, 0],
    #                            [0, -1, 0],
    #                            [0, 0, -1]])

    #         R_rot_z_270 = np.array([[0, 1, 0],
    #                                 [-1, 0, 0],
    #                                 [0, 0, 1]])

    #         R_final = R_rot_z_270 @ R_flip @ R_z

    #         T = np.eye(4)
    #         T[:3, :3] = R_final
    #         T[:3, 3] = np.asarray(tvec).reshape(3,)

    #         self.T_table_corners_to_camera_dict[f"corner_{marker_id}"] = T

    #     # Compute full base → camera transform for each corner
    #     self.T_base_to_camera_dict = {
    #         corner: self.T_base_to_corners_dict[corner] @ self.T_table_corners_to_camera_dict[corner]
    #         for corner in self.T_table_corners_to_camera_dict
    #         if corner in self.T_base_to_corners_dict
    #     }

    def get_average_transform_to_camera(self) -> np.ndarray:
        """
        Compute average base→camera transformation from multiple corner observations.
        Returns:
            np.ndarray: 4x4 averaged transformation matrix.
        """
        translations = []
        rotations = []

        for T in self.T_base_to_camera_dict.values():
            translations.append(T[:3, 3])
            rotations.append(R.from_matrix(T[:3, :3]))

        avg_translation = np.mean(translations, axis=0)
        avg_rotation = R.from_quat([r.as_quat() for r in rotations]).mean()

        T_avg = np.eye(4)
        T_avg[:3, :3] = avg_rotation.as_matrix()
        T_avg[:3, 3] = avg_translation
        return T_avg

    def get_transformation_matrix_from_point(self, point: PointWithOrientation) -> np.ndarray:
        """
        Get transformation matrix from a point with orientation to the base frame.
        
        Input:
            point (PointWithOrientation): 3D point with orientation
        Output:
            np.ndarray: 4x4 transformation matrix
        """
        # Create rotation matrix from roll, pitch, yaw
        rotation = R.from_euler('xyz', [point.roll, point.pitch, point.yaw]).as_matrix()

        # Create transformation matrix
        T = np.eye(4)
        T[:3, :3] = rotation
        T[:3, 3] = [point.x, point.y, point.z]

        return T

    def get_transform(self, source_frame, target_frame):
        """
        Get transformation matrix between two frames.
        Input:
            source_frame (str): 'base', 'camera', 'table', 'object'
            target_frame (str): 'base', 'camera', 'table', 'object'
        Output:
            np.ndarray: 4x4 transformation matrix
        """
        T = np.eye(4)
        if source_frame == 'camera' and target_frame == 'base':
            T = self.T_base_to_camera
        elif source_frame == 'base' and target_frame == 'camera':
            T = np.linalg.inv(self.T_base_to_camera)
        elif source_frame == 'table' and target_frame == 'base':
            T = self.T_base_to_corners_dict["corner_0"]
        else:
            print("?????????????????????????????????????")

        return T

    def update_object_pose(self, object_pose_camera):
        """
        Update object pose in camera frame.
        Input:
            object_pose_camera: [x, y, z, roll, pitch, yaw] in camera frame
        """
        position = object_pose_camera[:3]
        rotation = R.from_euler('xyz', object_pose_camera[3:]).as_matrix()
        
        self.T_camera_to_object[:3, :3] = rotation
        self.T_camera_to_object[:3, 3] = position

    def transform_point(self, point: PointWithOrientation, source_frame, target_frame) -> PointWithOrientation:
        """
        Transform a point from one frame to another.
        
        Input:
            point (PointWithOrientation): 3D point with orientation
            source_frame (str): 'base', 'camera', 'table', 'object'
            target_frame (str): 'base', 'camera', 'table', 'object'
        
        Output:
            PointWithOrientation: Transformed point in target frame
        """        
        # Get the transformation matrix
        T = self.get_transform(source_frame, target_frame)

        # Transform position
        position = np.array([point.x, point.y, point.z, 1.0])  # Convert to homogeneous coordinates
        transformed_position = T @ position
        new_position = transformed_position[:3]  # Extract x, y, z coordinates

        # Transform orientation using rotation matrix
        rotation = R.from_euler('xyz', [point.roll, point.pitch, point.yaw]).as_matrix()  # Convert to matrix
        transformed_rotation = T[:3, :3] @ rotation  # Apply rotation part of transform
        new_orientation = R.from_matrix(transformed_rotation).as_euler('xyz')  # Convert back to Euler angles

        # Create a new PointWithOrientation object with transformed data
        return PointWithOrientation(
            new_position[0], new_position[1], new_position[2] + self.z_calibration_constant,
            new_orientation[0], new_orientation[1], new_orientation[2]
        )

    def visusalise_environment(self, T_additional_to_visualise_dict: dict = None):
        """
        Visualize the environment with:
        - Table corners (base frame)
        - Estimated camera position
        - Coordinate axes at each corner

        Args:
            T_additional_to_visualise_dict - transformations to include in the visualisation {tf_name, tf} 
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("Environment Overview")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")

        # Plot additional T
        for name, T_corner in T_additional_to_visualise_dict.items():
            x, y, z = T_corner[:3, 3]
            ax.scatter(x, y, z, color='blue')
            ax.text(x, y, z + 0.01, name, color='blue')

            # Plot small axes to indicate orientation
            for i, color in zip(range(3), ['r', 'g', 'b']):
                end = T_corner[:3, 3] + 0.15 * T_corner[:3, i]
                ax.plot([x, end[0]], [y, end[1]], [z, end[2]], color=color)

        # Plot table corners
        for name, T_corner in self.T_base_to_corners_dict.items():
            x, y, z = T_corner[:3, 3]
            ax.scatter(x, y, z, color='blue')
            ax.text(x, y, z + 0.01, name, color='blue')

            # Plot small axes to indicate orientation
            for i, color in zip(range(3), ['r', 'g', 'b']):
                end = T_corner[:3, 3] + 0.15 * T_corner[:3, i]
                ax.plot([x, end[0]], [y, end[1]], [z, end[2]], color=color)

        # Plot estimated camera pose
        T_cam = self.T_base_to_camera
        x, y, z = T_cam[:3, 3]
        ax.scatter(x, y, z, color='orange', s=100, label="Camera")
        ax.text(x, y, z + 0.01, "camera", color='orange')

        # Draw camera frame axes
        for i, color in zip(range(3), ['r', 'g', 'b']):
            end = T_cam[:3, 3] + 0.15 * T_cam[:3, i]
            ax.plot([x, end[0]], [y, end[1]], [z, end[2]], color=color)

        ax.legend()
        ax.view_init(elev=25, azim=45)
        # Optional: set equal aspect ratio manually
        max_range = np.array([
            ax.get_xlim3d(), 
            ax.get_ylim3d(), 
            ax.get_zlim3d()
        ]).ptp(axis=1).max() / 2.0

        mid_x = np.mean(ax.get_xlim3d())
        mid_y = np.mean(ax.get_ylim3d())
        mid_z = np.mean(ax.get_zlim3d())

        ax.set_xlim3d(mid_x - max_range, mid_x + max_range)
        ax.set_ylim3d(mid_y - max_range, mid_y + max_range)
        ax.set_zlim3d(mid_z - max_range, mid_z + max_range)

        # Draw base frame at origin
        origin = np.array([0, 0, 0])
        ax.quiver(*origin, 0.1, 0, 0, color='r', linewidth=2)
        ax.quiver(*origin, 0, 0.1, 0, color='g', linewidth=2)
        ax.quiver(*origin, 0, 0, 0.1, color='b', linewidth=2)
        ax.text(0.12, 0, 0, "Base X", color='r')
        ax.text(0, 0.12, 0, "Base Y", color='g')
        ax.text(0, 0, 0.12, "Base Z", color='b')

        # Add legend for axis colors
        red_patch = mpatches.Patch(color='r', label='X axis')
        green_patch = mpatches.Patch(color='g', label='Y axis')
        blue_patch = mpatches.Patch(color='b', label='Z axis')
        ax.legend(handles=[red_patch, green_patch, blue_patch])

        # Ensure equal aspect
        max_range = np.array([
            ax.get_xlim3d(),
            ax.get_ylim3d(),
            ax.get_zlim3d()
        ]).ptp(axis=1).max() / 2.0

        mid_x = np.mean(ax.get_xlim3d())
        mid_y = np.mean(ax.get_ylim3d())
        mid_z = np.mean(ax.get_zlim3d())

        ax.set_xlim3d(mid_x - max_range, mid_x + max_range)
        ax.set_ylim3d(mid_y - max_range, mid_y + max_range)
        ax.set_zlim3d(mid_z - max_range, mid_z + max_range)

        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")

        plt.tight_layout()
        plt.show()
