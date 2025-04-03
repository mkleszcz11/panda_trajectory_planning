import rospy
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint, JointTrajectory

rospy.init_node("send_trajectory")

client = actionlib.SimpleActionClient("/position_joint_trajectory_controller/follow_joint_trajectory", FollowJointTrajectoryAction)
client.wait_for_server()

goal = FollowJointTrajectoryGoal()
goal.trajectory = JointTrajectory()
goal.trajectory.joint_names = [
    "panda_joint1", "panda_joint2", "panda_joint3",
    "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7"
]

point = JointTrajectoryPoint()
point.positions = [0.0, 0.3, 0.1, -2.0, 0.0, 2.0, 0.5]
point.time_from_start = rospy.Duration(2.0)
goal.trajectory.points.append(point)

client.send_goal(goal)
client.wait_for_result()