
import math

class PointWithOrientation:
    """
    Class representing a point in 3D space with orientation.
    """
    def __init__(self, x=0, y=0, z=0, roll=0, pitch=math.pi-0.01, yaw=0):
        self.x = x
        self.y = y
        self.z = z
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
    
    def set_position(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def set_orientation(self, roll, pitch, yaw):
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw

    def set_position_and_orientation(self, x, y, z, roll, pitch, yaw):
        self.set_position(x, y, z)
        self.set_orientation(roll, pitch, yaw)
        
    def get_position(self):
        return [self.x, self.y, self.z]
    
    def get_orientation(self):
        return [self.roll, self.pitch, self.yaw]
    
    def get_position_and_orientation_as_list(self) -> list:
        return [self.x, self.y, self.z, self.roll, self.pitch, self.yaw]


class PandaTransformations:
    """
    Class allowing to transform between different frames of the setup.
    Main frames are:
    - Panda base link (world frame)
    - Panda end effector - computed with forward kinematics.
    - Camera frame - calibrated based on aruco markers in the table corners. Camera looks down on the table.
    - Point (0,0,0) on a table, only translation in relation to the base link - calibrated manually. Translation values to 4 corners are hardcoded.
    - Object frame - object is placed on the table, its position is computed based on the camera frame.

    Class allows to transform between all of the frames.
    """
    def __init__(self):
        self.table_corners = {
            "top_right": PointWithOrientation(0.6866172576844853, -0.40264805670207643, 0.1882167138629311), # this is the point (0,0,0) on the table / marker 0
            "top_left": PointWithOrientation(0.7048439047888219, 0.3948503398027187, 0.19247876477108974), # marker 1 TODO
            "bottom_right": PointWithOrientation(0.09612144178867847, 0.41001726589279464, 0.19518941212650254), # marker 2 TODO
            "bottom_left": PointWithOrientation(0.09329019345166721, -0.4108675959286945, 0.19623268710457104) # marker 3 TODO
        }

    def transform_table_to_base_link(self, point: PointWithOrientation) -> PointWithOrientation:
        """
        Transform a point on the table to the base link frame.
        It should allow to compute the position of the object in the base link frame.

        Example:
            Object lays on a table in point [0,0] (table corner). By calling this function and passing the
            object position as an argument, we can compute the object position in the base link frame and pass
            it directly to the robot controller (move_to_point function).
        """
        transformed_point = PointWithOrientation()
        
        transformed_point.set_position(point.x + self.table_corners["top_right"].x,
                                       point.y + self.table_corners["top_right"].y,
                                       point.z + self.table_corners["top_right"].z)
        
        transformed_point.set_orientation(point.roll, point.pitch, point.yaw)

        return transformed_point
        
        
        

        
        
        
