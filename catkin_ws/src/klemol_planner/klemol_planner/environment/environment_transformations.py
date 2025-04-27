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
    def __init__(self, cam_operations: CameraOperations) -> None:
        # Table corners in base frame
        # Found with command:
        # rosrun tf tf_echo /panda_link0 /panda_link8
        self.camera_operations = cam_operations

        # This is the height from flange (last coordinate system) to the table while we are calibrating
        self.z_calibration_constant = 0.0#0.12

        self.table_corners_translations = {
            "corner_0": np.array([0.687, -0.385, 0.177]),# - self.z_calibration_constant]),
            "corner_1": np.array([0.674,  0.366, 0.183]),# - self.z_calibration_constant]),
            "corner_2": np.array([0.100,  0.412, 0.195]),# - self.z_calibration_constant]),
            "corner_3": np.array([0.099, -0.412, 0.196])# - self.z_calibration_constant])
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
        Calibrate the camera by finding its position and orientation relative to the base.

        Solve everything in 2D. In last steps Z dimension is added and self.T_base_to_camera is defined.

        NOTE: If you are some lucky student in NRT lab working with this code, you might be wondering,
        isn't there a better way to do this? Probably there is, but be prepared to spend substantial
        amount of time and get the same result.

        Steps:
        1. Get the (x, y) positions of all corners relative to the base frame.
        2. Get the (x, y) positions of all corners relative to the camera frame.
        (Note: Camera sees a mirror image because its z-axis points downward - make a mirror along X axis)
        - Keep corners ordered counterclockwise.
        - Camera frame must match base frame orientation: y points up (C1C2 edge), x points right (C0C1 edge).
        - After calibration, as the last step, we will flip the camera transformation along the X-axis again.

        [VISUALIZE] Show corners separately for base and camera. Mark centroids and (0,0) axes (y-up, x-right).

        3. Calculate the centroid (mean) of base_corners_xy → mean_in_base_frame.
        4. Calculate the centroid (mean) of camera_corners_xy → mean_in_camera_frame.

        [VISUALIZE] Treat both centroids as (0, 0). Plot both sets of points on the same plot to verify alignment.

        5. Use the Kabsch algorithm to find the rotation angle (camera_to_base_rotation) between the two sets.

        6. Translate camera_corners_xy and the camera frame by:
        camera_translation = mean_in_base_frame - mean_in_camera_frame.
        (Check: After translation, the centroids must overlap exactly.)

        [VISUALIZE] Show both point sets on one plot. Show origins and centroids (they should overlap).

        7. Rotate the translated camera_corners_xy and camera origin using camera_to_base_rotation.

        [VISUALIZE] Show updated points and frames after rotation.

        8. Calculate cam_z:
        For each corner, add its distance from the camera (camera z) to its z in the base frame.
        Take the average of all results to get cam_z.

        9. Build T_base_to_camera_before_flip.
            Apply an X-axis flip using X_flip_matrix.

        10. Save the final transformation as self.T_base_to_camera.
        """
        print("[INFO] Calibrating camera...")
        camera_op = self.camera_operations
        detected_markers = camera_op.get_marker_transforms()

        if len(detected_markers) < 4:
            raise RuntimeError("Not all 4 corners were detected")

        base_corners_xy = []
        camera_corners_xy = []
        cam_zs = []

        base_xy = np.array([[0.0], [0.0]])
        camera_xy = np.array([[0.0], [0.0]])
        camera_rotation = np.array([[1.0, 0.0], [0.0, 1.0]]) # Identity matrix

        for corner_name, T_cam_marker in detected_markers.items():
            if corner_name not in self.table_corners_translations:
                continue
            p_cam = T_cam_marker[:3, 3]
            p_base = self.table_corners_translations[corner_name]

            camera_corners_xy.append(p_cam[:2])
            base_corners_xy.append(p_base[:2])
            cam_zs.append(p_cam[2] + p_base[2])

        base_corners_xy = np.array(base_corners_xy).T  # shape (2, N) -> np.array([[x1, x2, x3, x4], [y1, y2, y3, y4]])
        camera_corners_xy = np.array(camera_corners_xy).T

        # Flip X-axis only
        camera_corners_xy[0, :] *= -1

        # Compute centroids
        base_centroid = np.mean(base_corners_xy, axis=1, keepdims=True)
        camera_centroid = np.mean(camera_corners_xy, axis=1, keepdims=True)

        # VISUALIZE STEP 1
        self.__visualize_corners(base_corners_xy=base_corners_xy,
                                 camera_corners_xy=camera_corners_xy,
                                 base_centroid=base_centroid,
                                 camera_centroid=camera_centroid,
                                 base_origin=base_xy,
                                 camera_origin=camera_xy,
                                 rotation=camera_rotation,
                                 mode='separate',
                                 plot_title="Step 1: Before Centering")

        # Center the point clouds
        base_corners_xy_centered = base_corners_xy - base_centroid
        camera_corners_xy_centered = camera_corners_xy - camera_centroid

        base_centroid_after_centering = np.mean(base_corners_xy_centered, axis=1, keepdims=True)
        camera_centroid_after_centering = np.mean(camera_corners_xy_centered, axis=1, keepdims=True)

        base_xy_after_centering = base_xy - base_centroid
        camera_xy_after_centering = camera_xy - camera_centroid
        print(f"BASE XY AFTER CENTERING: {base_xy_after_centering}")
        print(f"CAMERA XY AFTER CENTERING: {camera_xy_after_centering}")

        # VISUALIZE STEP 2
        self.__visualize_corners(
            base_corners_xy=base_corners_xy_centered,
            camera_corners_xy=camera_corners_xy_centered,
            base_centroid=base_centroid_after_centering,
            camera_centroid=camera_centroid_after_centering,
            base_origin=base_xy_after_centering,
            camera_origin=camera_xy_after_centering,
            rotation=camera_rotation,
            mode='combined',
            plot_title="Step 2: After Centering"
        )

        # STEP 3: Find optimal rotation using Kabsch algorithm

        H = camera_corners_xy_centered @ base_corners_xy_centered.T  # (2xN) @ (Nx2).T → (2x2)

        U, _, Vt = np.linalg.svd(H)
        R_2D = Vt.T @ U.T

        # Ensure a right-handed coordinate system
        if np.linalg.det(R_2D) < 0:
            Vt[1, :] *= -1
            R_2D = Vt.T @ U.T
        # Apply rotation
        camera_corners_xy_rotated = R_2D @ camera_corners_xy_centered
        camera_xy_rotated = R_2D @ camera_xy_after_centering

        # VISUALIZE STEP 3
        self.__visualize_corners(
            base_corners_xy=base_corners_xy_centered,
            camera_corners_xy=camera_corners_xy_rotated,
            base_centroid=base_centroid_after_centering,
            camera_centroid=camera_centroid_after_centering,
            base_origin=base_xy_after_centering,
            camera_origin=camera_xy_rotated,
            rotation=camera_rotation,
            mode='combined',
            plot_title="Step 3: After Rotation"
        )

        # STEP 4: Calculate camera Z and flip entire transformation
        # Estimate Z
        x_translation = camera_xy_rotated[0, 0] - base_xy_after_centering[0, 0]
        y_translation = camera_xy_rotated[1, 0] - base_xy_after_centering[1, 0]
        z_translation = np.mean(cam_zs)

        print(f"X translation: {x_translation:.4f} m")
        print(f"Y translation: {y_translation:.4f} m")
        print(f"Z translation: {z_translation:.4f} m")

        print(f"camera_xy_rotated: {camera_xy_rotated}")

        # 5. Build T_base_to_camera_before_flip
        T_base_to_camera_before_flip = np.array([
            [R_2D[0, 0], R_2D[0, 1], 0, x_translation],
            [R_2D[1, 0], R_2D[1, 1], 0, y_translation],
            [0, 0, 1, z_translation],
            [0, 0, 0, 1]
        ])

        print(f"[INFO] T_base_to_camera_before_flip:\n{np.round(T_base_to_camera_before_flip, 4)}")

        # Apply X-axis flip correction
        X_flip = np.array([[1,  0,  0, 0],
                           [0, -1,  0, 0],
                           [0,  0, -1, 0],
                           [0,  0,  0, 1]])

        # For whatever reason we also have to flip around new Z axis
        Z_flip = np.array([[-1,  0, 0, 0],
                           [ 0, -1, 0, 0],
                           [ 0,  0, 1, 0],
                           [ 0,  0, 0, 1]])

        self.T_base_to_camera = T_base_to_camera_before_flip @ X_flip @ Z_flip

        print(f"[INFO] Final T_base_to_camera (after X flip):\n{np.round(self.T_base_to_camera, 4)}")


    def __visualize_corners(self, base_corners_xy: np.ndarray, camera_corners_xy: np.ndarray,
                            base_centroid: np.ndarray, camera_centroid: np.ndarray,
                            base_origin: np.ndarray = np.array([[0], [0]]),
                            camera_origin: np.ndarray = np.array([[0], [0]]),
                            rotation: np.ndarray = None,
                            mode: str = 'separate',
                            plot_title: str = "Camera Calibration") -> None:
        """
        Helper to visualize the environment.

        Args:
            base_corners_xy: (2, N) true base corners
            camera_corners_xy: (2, N) detected camera corners (already flipped X)
            base_centroid: (2,1)
            camera_centroid: (2,1)
            base_origin: (2,1) base frame origin position
            camera_origin: (2,1) camera frame origin position
            rotation: (2,2) rotation matrix (optional)
            mode: 'separate' or 'combined'
        """
        fig = plt.figure(figsize=(8, 6))
        if mode == 'separate':
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            fig.suptitle(plot_title)

            # --- Base Frame ---
            ax1.set_title("Base Frame")
            ax1.scatter(base_corners_xy[0], base_corners_xy[1], c='blue', label='Base Corners')
            ax1.scatter(*base_centroid, c='black', marker='x', label='Base Centroid')

            for i in range(base_corners_xy.shape[1]):
                ax1.text(base_corners_xy[0, i], base_corners_xy[1, i], f"B{i}", color='blue')

            # Draw base axes at base_origin
            ax1.quiver(*base_origin.flatten(), 0.1, 0, color='red', angles='xy', scale_units='xy', scale=1, label='Base X')
            ax1.quiver(*base_origin.flatten(), 0, 0.1, color='green', angles='xy', scale_units='xy', scale=1, label='Base Y')

            ax1.set_aspect('equal')
            ax1.grid(True)
            ax1.legend()
            ax1.set_xlabel("X [m]")
            ax1.set_ylabel("Y [m]")

            # --- Camera Frame ---
            ax2.set_title("Camera Frame")
            cam = camera_corners_xy.copy()

            if rotation is not None:
                cam = rotation @ cam

            ax2.scatter(cam[0], cam[1], c='orange', label='Camera Corners')
            ax2.scatter(*camera_centroid, c='black', marker='x', label='Camera Centroid')

            for i in range(cam.shape[1]):
                ax2.text(cam[0, i], cam[1, i], f"C{i}", color='orange')

            # Draw camera axes at camera_origin, rotated
            if rotation is not None:
                camera_x_axis = rotation @ np.array([0.1, 0]).reshape(2, 1)
                camera_y_axis = rotation @ np.array([0, 0.1]).reshape(2, 1)
            else:
                camera_x_axis = np.array([[0.1], [0]])
                camera_y_axis = np.array([[0], [0.1]])

            ax2.quiver(*camera_origin.flatten(), camera_x_axis[0, 0], camera_x_axis[1, 0], color='r', angles='xy', scale_units='xy', scale=1, label='Cam X')
            ax2.quiver(*camera_origin.flatten(), camera_y_axis[0, 0], camera_y_axis[1, 0], color='g', angles='xy', scale_units='xy', scale=1, label='Cam Y')

            ax2.set_aspect('equal')
            ax2.grid(True)
            ax2.legend()
            ax2.set_xlabel("X [m]")
            ax2.set_ylabel("Y [m]")

            plt.tight_layout()

        elif mode == 'combined':
            ax = fig.add_subplot(111)
            fig.suptitle(plot_title)

            ax.scatter(base_corners_xy[0], base_corners_xy[1], c='blue', label='Base Corners')
            ax.scatter(*base_centroid, c='blue', marker='x', label='Base Centroid', s=50)

            cam = camera_corners_xy.copy()
            if rotation is not None:
                cam = rotation @ cam

            ax.scatter(cam[0], cam[1], c='orange', label='Camera Corners')
            ax.scatter(*camera_centroid, c='none', edgecolors='orange', marker='o', label='Camera Centroid', s=50, linewidths=2)

            for i in range(base_corners_xy.shape[1]):
                ax.text(base_corners_xy[0, i], base_corners_xy[1, i], f"B{i}", color='blue')

            for i in range(cam.shape[1]):
                ax.text(cam[0, i], cam[1, i], f"C{i}", color='orange')

            # Draw base axes
            ax.quiver(*base_origin.flatten(), 0.1, 0, color='red', label='Base X')
            ax.quiver(*base_origin.flatten(), 0, 0.1, color='green', label='Base Y')

            # Draw camera axes rotated
            if rotation is not None:
                camera_x_axis = rotation @ np.array([0.1, 0]).reshape(2, 1)
                camera_y_axis = rotation @ np.array([0, 0.1]).reshape(2, 1)
            else:
                camera_x_axis = np.array([[0.1], [0]])
                camera_y_axis = np.array([[0], [0.1]])

            ax.quiver(*camera_origin.flatten(), camera_x_axis[0, 0], camera_x_axis[1, 0], color='r', linestyle='dashed', label='Cam X')
            ax.quiver(*camera_origin.flatten(), camera_y_axis[0, 0], camera_y_axis[1, 0], color='g', linestyle='dashed', label='Cam Y')

            ax.set_aspect('equal')
            ax.grid(True)
            ax.legend()
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Y [m]")
            plt.tight_layout()

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

    def visusalise_environment(self, T_additional_to_visualise_dict: dict = None, show_points_orientation: bool = True) -> None:
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
                if show_points_orientation:
                    ax.plot([x, end[0]], [y, end[1]], [z, end[2]], color=color)

        # Plot table corners
        for name, T_corner in self.T_base_to_corners_dict.items():
            x, y, z = T_corner[:3, 3]
            ax.scatter(x, y, z, color='blue')
            ax.text(x, y, z + 0.01, name, color='blue')

            # Plot small axes to indicate orientation
            for i, color in zip(range(3), ['r', 'g', 'b']):
                end = T_corner[:3, 3] + 0.15 * T_corner[:3, i]
                if show_points_orientation:
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
