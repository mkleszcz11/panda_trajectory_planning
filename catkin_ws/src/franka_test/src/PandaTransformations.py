
import numpy as np
from scipy.spatial.transform import Rotation as R
from PointWithOrientation import PointWithOrientation

class PandaTransformations:
    """
    Class allowing to transform between different frames of the setup.
    Main frames are:
    - Panda base link (world frame)
    - Panda end effector - computed with forward kinematics.
    - Camera frame - calibrated based on aruco markers in the table corners. Camera looks down on the table.
    - Point (0,0,0) on a table, only translation in relation to the base link - calibrated manually. Translation values to 4 corners are hardcoded.
    - Object frame - object is placed on the table, its position is computed based on the camera frame.
    """
    def __init__(self):
        # Table corners in base frame
        self.table_corners = {
            "top_right": np.array([0.6866172576844853, -0.40264805670207643, 0.1882167138629311]),
            "top_left": np.array([0.7048439047888219, 0.3948503398027187, 0.19247876477108974]),
            "bottom_right": np.array([0.09612144178867847, 0.41001726589279464, 0.19518941212650254]),
            "bottom_left": np.array([0.09329019345166721, -0.4108675959286945, 0.19623268710457104])
        }
        
        # Transformation matrices (initialized to identity)
        self.T_base_to_table = np.array([
            [0, -1, 0, self.table_corners["top_right"][0]],
            [1, 0, 0, self.table_corners["top_right"][1]],
            [0, 0, 1, self.table_corners["top_right"][2]],
            [0, 0, 0, 1]
        ])
        self.T_base_to_camera = np.eye(4)
        self.T_camera_to_object = np.eye(4)

    def calibrate_camera(self, aruco_marker_positions):
        """
        Calibrate camera position based on known ArUco marker positions.
        Input:
            aruco_marker_positions: dict
                Dictionary with marker IDs as keys and 3D positions in camera frame as values.
                Example: {0: [x, y, z], 1: [x, y, z], ...}
        Output:
            np.ndarray: 4x4 transformation matrix from camera to table frame.
        """
        if len(aruco_marker_positions) != 4:
            raise ValueError("Exactly 4 ArUco markers are required for calibration.")
        
        # Compute the transformation matrix using point-to-point alignment
        camera_points = np.array(list(aruco_marker_positions.values()))
        base_points = np.array(list(self.table_corners.values()))
        
        # Solve for transformation using Kabsch algorithm
        H = camera_points.T @ base_points
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        t = base_points.mean(axis=0) - R @ camera_points.mean(axis=0)

        T_camera_to_table = np.eye(4)
        T_camera_to_table[:3, :3] = R
        T_camera_to_table[:3, 3] = t
        
        self.T_base_to_camera = np.linalg.inv(T_camera_to_table)
        return self.T_base_to_camera

    def get_transform(self, source_frame, target_frame):
        """
        Get transformation matrix between two frames.
        Input:
            source_frame (str): 'base', 'camera', 'table', 'object'
            target_frame (str): 'base', 'camera', 'table', 'object'
        Output:
            np.ndarray: 4x4 transformation matrix
        """
        transforms = {
            'base': np.eye(4),
            'camera': self.T_base_to_camera,
            'table': self.T_base_to_table,
            'object': self.T_camera_to_object @ self.T_base_to_camera
        }
        if source_frame not in transforms or target_frame not in transforms:
            raise ValueError(f"Invalid frame name: {source_frame}, {target_frame}")
        print(f"---- Source frame: {source_frame}, Target frame: {target_frame}")
        # Compute transformation matrix from source to target frame.
        T_source_to_target = np.linalg.inv(transforms[target_frame]) @ transforms[source_frame]
        print(f"---- T_source_to_target:\n{T_source_to_target}")
        return T_source_to_target

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
        print(f"Transforming point -> {point} from {source_frame} to {target_frame}")
        
        # Get the transformation matrix
        T = self.get_transform(source_frame, target_frame)
        print(f"---- T:\n{T}")

        # Transform position
        position = np.array([point.x, point.y, point.z, 1.0])  # Convert to homogeneous coordinates
        transformed_position = T @ position
        new_position = transformed_position[:3]  # Extract x, y, z coordinates

        print(f"Transformed position: {new_position}")

        # Transform orientation using rotation matrix
        rotation = R.from_euler('xyz', [point.roll, point.pitch, point.yaw]).as_matrix()  # Convert to matrix
        transformed_rotation = T[:3, :3] @ rotation  # Apply rotation part of transform
        new_orientation = R.from_matrix(transformed_rotation).as_euler('xyz')  # Convert back to Euler angles

        print(f"Transformed orientation (roll, pitch, yaw): {new_orientation}")

        # Create a new PointWithOrientation object with transformed data
        return PointWithOrientation(
            new_position[0], new_position[1], new_position[2],
            new_orientation[0], new_orientation[1], new_orientation[2]
        )

        

# Example usage:
if __name__ == "__main__":
    panda_tf = PandaTransformations()
    
    # Example ArUco marker calibration data (in camera frame)
    aruco_positions = {
        0: [0.1, 0.2, 0.3],
        1: [0.4, 0.2, 0.3],
        2: [0.1, -0.2, 0.3],
        3: [-0.1, -0.2, 0.3]
    }
    
    T_base_to_camera = panda_tf.calibrate_camera(aruco_positions)
    print("Camera Calibration Result:\n", T_base_to_camera)

    # Update object pose from camera frame
    panda_tf.update_object_pose([0.5, 0.1, 0.2, 0, 0, 0])
    
    # Transform point from object to base frame
    point = np.array([0.1, 0.2, 0.3])
    transformed_point = panda_tf.transform_point(point, 'object', 'base')
    print("Transformed Point:\n", transformed_point)








# import math

# class PointWithOrientation:
#     """
#     Class representing a point in 3D space with orientation.
#     """
#     def __init__(self, x=0, y=0, z=0, roll=0, pitch=math.pi-0.01, yaw=0):
#         self.x = x
#         self.y = y
#         self.z = z
#         self.roll = roll
#         self.pitch = pitch
#         self.yaw = yaw
    
#     def set_position(self, x, y, z):
#         self.x = x
#         self.y = y
#         self.z = z

#     def set_orientation(self, roll, pitch, yaw):
#         self.roll = roll
#         self.pitch = pitch
#         self.yaw = yaw

#     def set_position_and_orientation(self, x, y, z, roll, pitch, yaw):
#         self.set_position(x, y, z)
#         self.set_orientation(roll, pitch, yaw)
        
#     def get_position(self):
#         return [self.x, self.y, self.z]
    
#     def get_orientation(self):
#         return [self.roll, self.pitch, self.yaw]
    
#     def get_position_and_orientation_as_list(self) -> list:
#         return [self.x, self.y, self.z, self.roll, self.pitch, self.yaw]


# class PandaTransformations:
#     """
#     Class allowing to transform between different frames of the setup.
#     Main frames are:
#     - Panda base link (world frame)
#     - Panda end effector - computed with forward kinematics.
#     - Camera frame - calibrated based on aruco markers in the table corners. Camera looks down on the table.
#     - Point (0,0,0) on a table, only translation in relation to the base link - calibrated manually. Translation values to 4 corners are hardcoded.
#     - Object frame - object is placed on the table, its position is computed based on the camera frame.

#     Class allows to transform between all of the frames.
#     """
#     def __init__(self):
#         self.table_corners = {
#             "top_right": PointWithOrientation(0.6866172576844853, -0.40264805670207643, 0.1882167138629311), # this is the point (0,0,0) on the table / marker 0
#             "top_left": PointWithOrientation(0.7048439047888219, 0.3948503398027187, 0.19247876477108974), # marker 1 TODO
#             "bottom_right": PointWithOrientation(0.09612144178867847, 0.41001726589279464, 0.19518941212650254), # marker 2 TODO
#             "bottom_left": PointWithOrientation(0.09329019345166721, -0.4108675959286945, 0.19623268710457104) # marker 3 TODO
#         }

#         self.matrix_camera_to_base_link =

#     def transform_table_to_base_link(self, point: PointWithOrientation) -> PointWithOrientation:
#         """
#         Transform a point on the table to the base link frame.
#         It should allow to compute the position of the object in the base link frame.

#         Example:
#             Object lays on a table in point [0,0] (table corner). By calling this function and passing the
#             object position as an argument, we can compute the object position in the base link frame and pass
#             it directly to the robot controller (move_to_point function).
#         """
#         transformed_point = PointWithOrientation()
        
#         transformed_point.set_position(point.x + self.table_corners["top_right"].x,
#                                        point.y + self.table_corners["top_right"].y,
#                                        point.z + self.table_corners["top_right"].z)
        
#         transformed_point.set_orientation(point.roll, point.pitch, point.yaw)

#         return transformed_point
        
        
        

        
        
        
