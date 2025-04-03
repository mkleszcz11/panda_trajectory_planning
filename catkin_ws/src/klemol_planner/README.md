Plan:

1. (DONE) Planning (e.g., RRTConnect) → Generates waypoint path (joint states).
2. (SKIP FOR NOW) Path Simplification → Removes redundant waypoints.
3. Time Parameterization → Assigns velocities/accelerations.
4. Execution → Sends trajectory to controller (for instance via FollowJointTrajectoryAction).
