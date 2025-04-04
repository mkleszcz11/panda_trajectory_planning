
import numpy as np
from scipy.spatial.transform import Rotation as R
from klemol_planner.goals.point_with_orientation import PointWithOrientation

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
        # Transformation from base to table frame
        self.T_base_to_table = np.array([
            [0, -1, 0, self.table_corners["top_right"][0]],
            [1, 0, 0, self.table_corners["top_right"][1]],
            [0, 0, 1, self.table_corners["top_right"][2]],
            [0, 0, 0, 1]
        ])

        # Transformation from camera to table frame (code 0 - top right corner)
        # It is used to compute the transformation from camera to base frame
        # temp_x_value =  - self.table_corners["top_right"][1] # in axis
        # temp_y_value = self.table_corners["top_right"][0] - 0.35 # as in launchfile
        # temp_z_value = 1.5 - self.table_corners["top_right"][2] # as in launchfile
        temp_x_value = 0.4508
        temp_y_value = 0.2158
        temp_z_value = 1.29
        self.T_table_to_camera = np.array([
            [1, 0, 0, temp_x_value],
            [0, -1, 0, temp_y_value],
            [0, 0, -1, temp_z_value],
            [0, 0, 0, 1]
        ])

        # T_base_to_camera = T_base_to_table * T_table_to_camera
        # T_table_to_camera = np.linalg.inv(self.T_camera_to_table)
        self.T_base_to_camera = self.T_base_to_table @ self.T_table_to_camera

        # Transformation from camera to object frame will be updated based on object pose
        self.T_camera_to_object = np.eye(4)

    def calibrate_camera(self) -> None:
        """
        #TODO ANDERS HERE
        Calibrate camera position based on ArUco marker positions.

        Assign 4x4 transformation matrix from table to camera (table frame = code 0 - top right corner)
        to self.T_table_to_camera.

        Returns:
            None
        """
        # DO SOME MAGIC HERE
        magic_x_value = 0.0
        magic_y_value = 0.0
        magic_z_value = 0.0
        self.T_table_to_camera = np.array([
            [-1, 0, 0, magic_x_value],
            [0, 1, 0, magic_y_value],
            [0, 0, -1, magic_z_value],
            [0, 0, 0, 1]
        ])

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
            new_position[0], new_position[1], new_position[2],
            new_orientation[0], new_orientation[1], new_orientation[2]
        )

        



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
        
        
        

        
        
        
