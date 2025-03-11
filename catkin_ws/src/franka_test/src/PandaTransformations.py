


class PointWithOrientation:
    """
    Class representing a point in 3D space with orientation.
    """
    def __init__(self, x, y, z):
        self.x = 0
        self.y = 0
        self.z = 0
        self.roll = 0
        self.pitch = 0
        self.yaw = 0

    def set_position(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def set_orientation(self, roll, pitch, yaw):
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        
    def get_position(self):
        return [self.x, self.y, self.z]
    
    def get_orientation(self):
        return [self.roll, self.pitch, self.yaw]


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
            "top_right": [0.5, 0.5, 0], # this is the point (0,0,0) on the table / marker 0
            "top_left": [0.5, -0.5, 0], # marker 1 TODO
            "bottom_right": [-0.5, -0.5, 0], # marker 2 TODO
            "bottom_left": [0.0, 0.5, 0] # marker 3 TODO
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
        
        transformed_point.set_position(point.x + self.table_corners["top_right"][0],
                                       point.y + self.table_corners["top_right"][1],
                                       point.z + self.table_corners["top_right"][2])

        return transformed_point
        
        
        

        
        
        
