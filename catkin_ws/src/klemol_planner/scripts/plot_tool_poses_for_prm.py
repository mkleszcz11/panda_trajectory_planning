import rospy
import numpy as np
import tf
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseArray, Pose
from urdf_parser_py.urdf import URDF
from kdl_parser_py.urdf import treeFromUrdfModel
import PyKDL as kdl
import argparse

def build_kdl_chain(urdf_path, base_link, tip_link):
    with open(urdf_path, 'r') as file:
        urdf_str = file.read()
    urdf_model = URDF.from_xml_string(urdf_str)
    ok, tree = treeFromUrdfModel(urdf_model)
    if not ok:
        raise RuntimeError("Failed to parse URDF into KDL tree")
    chain = tree.getChain(base_link, tip_link)
    return chain

def compute_fk(chain, joint_angles):
    fk_solver = kdl.ChainFkSolverPos_recursive(chain)
    jnt_array = kdl.JntArray(chain.getNrOfJoints())
    for i in range(len(joint_angles)):
        jnt_array[i] = joint_angles[i]
    frame = kdl.Frame()
    fk_solver.JntToCart(jnt_array, frame)
    return frame

def load_configs(path):
    data = np.load(path)
    return data['configs']

def pose_from_kdl(frame):
    position = frame.p
    rot = frame.M
    quat = rot.GetQuaternion()
    pose = Pose()
    pose.position.x = position[0]
    pose.position.y = position[1]
    pose.position.z = position[2]
    pose.orientation.x = quat[0]
    pose.orientation.y = quat[1]
    pose.orientation.z = quat[2]
    pose.orientation.w = quat[3]
    return pose

def main():
    rospy.init_node('tool_pose_visualizer')
    pub = rospy.Publisher('/tool_poses', PoseArray, queue_size=1, latch=True)

    urdf_path = rospy.get_param("~urdf_path")
    base_link = rospy.get_param("~base_link", "panda_link0")
    tip_link = rospy.get_param("~tip_link", "panda_link8")
    prm_path = rospy.get_param("~prm_path")

    chain = build_kdl_chain(urdf_path, base_link, tip_link)
    configs = load_configs(prm_path)

    pose_array = PoseArray()
    pose_array.header.stamp = rospy.Time.now()
    pose_array.header.frame_id = base_link

    for q in configs:
        frame = compute_fk(chain, q)
        pose_array.poses.append(pose_from_kdl(frame))

    pub.publish(pose_array)
    rospy.loginfo("Published {} tool poses".format(len(pose_array.poses)))
    rospy.spin()

if __name__ == '__main__':
    main()
