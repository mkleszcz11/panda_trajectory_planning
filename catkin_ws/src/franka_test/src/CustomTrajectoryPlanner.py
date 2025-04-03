class CustomTrajectoryPlanner:
    def __init__(self, robot):
        self.robot = robot
        self.current_position = robot.get_current_position()
        self.target_position = None
        self.trajectory = []

    def set_target_position(self, target_position):
        self.target_position = target_position

    def plan_trajectory(self):
        if self.target_position is None:
            raise ValueError("Target position not set.")
        
        # Simple linear trajectory for demonstration
        self.trajectory = [
            (self.current_position[0] + i * (self.target_position[0] - self.current_position[0]) / 10,
             self.current_position[1] + i * (self.target_position[1] - self.current_position[1]) / 10,
             self.current_position[2] + i * (self.target_position[2] - self.current_position[2]) / 10)
            for i in range(11)
        ]

    def execute_trajectory(self):
        for position in self.trajectory:
            self.robot.move_to(position)