# NOTE
# This code alongside normal PRM is an abomination, we
# are running out of time for thesis implementation,
# so if you are reading this, I am sorry :)

import numpy as np
import typing as t
from scipy.spatial import cKDTree
from concurrent.futures import ThreadPoolExecutor
from klemol_planner.environment.robot_model import Robot
from klemol_planner.environment.collision_checker import CollisionChecker
from klemol_planner.planners.nodes import GraphNode
from klemol_planner.planners.base import Planner
from klemol_planner.planners.nodes import GraphNode
import os
from klemol_planner.planners.base import Planner
from klemol_planner.planners.nodes import GraphNode
import moveit_commander
from geometry_msgs.msg import PoseStamped
import rospy

class PRMRoadmapBuilder:
    """
    Builds a roadmap for a PRM planner
    """
    def __init__(self,
                    robot_model,
                    collision_checker,
                    n_nodes: int = 1000,
                    k_neighbors: int = 5,
                    weights: list = [1.0]*7,
                    path_to_save: str = "prm_roadmap.npz",
                    restrict_to_task_space: bool = False,
                    limit_tool_orientation: bool = False):

        self.robot_model = robot_model
        self.collision_checker = collision_checker
        self.n_nodes = n_nodes
        self.k_neighbors = k_neighbors
        self.weights = weights
        self.path_to_save = path_to_save
        self.restrict_to_task_space = restrict_to_task_space
        self.limit_tool_orientation = limit_tool_orientation
        self.roadmap = {}
        self.kdtree = None
        self.next_node_id = 0

    def build_roadmap(self) -> None:
        """
        Samples valid nodes, connects neighbors using KDTree + collision check
        """
        print(f"--- building roadmap ---")
        self.roadmap.clear()
        valid_configs = []

        self.scene = moveit_commander.PlanningSceneInterface()
        self.add_box_obstacle(
            name="inflated_stick",
            size=(0.02, 0.02, 0.8),
            position=(0.42, 0.0, 0.4),
            collision_margin=0.03  # Add 3 cm clearance on all sides
        )

        i = 0
        while len(valid_configs) < self.n_nodes:
            if i % 1000 == 0:
                print(f"Sampling node {len(valid_configs)}/{self.n_nodes}")
            if self.restrict_to_task_space and not self.limit_tool_orientation:
                q = self.robot_model.sample_random_configuration_in_task_space()
            elif self.restrict_to_task_space and self.limit_tool_orientation:
                q = self.robot_model.sample_random_configuration_in_task_space_with_limited_tool_orientation()
            else:
                q = self.robot_model.sample_random_configuration()
            if self.robot_model.is_within_limits(q):# and not self.collision_checker.is_in_collision(q):
                valid_configs.append(q)
            i += 1

        for i, config in enumerate(valid_configs):
            if i % 1000 == 0:
                print(f"Adding node {i}/{self.n_nodes}")
            self._add_node(config)

        configs = np.array([node.config for node in self.roadmap.values()])
        self.kdtree = cKDTree(configs)

        # for node in self.roadmap.values():
        #     print(f"Connecting node {node.id}/{len(self.roadmap)}")
        #     distances, indices = self.kdtree.query(node.config, k=self.k_neighbors + 1)
        #     for idx in indices[1:]:  # skip self
        #         neighbor = list(self.roadmap.values())[idx]
        #         if neighbor.id not in node.edges:
        #             if self._is_collision_free_path(node.config, neighbor.config):
        #                 cost = self._weighted_distance(node.config, neighbor.config)
        #                 node.add_edge(neighbor.id, cost)
        #                 neighbor.add_edge(node.id, cost)

        self._connect_nodes_parallel()

    def _connect_nodes_parallel(self):
        print(f"--- connecting roadmap nodes (parallel) ---")
        node_list = list(self.roadmap.values())
        all_pairs = []

        # Generate connection candidates (only forward edges)
        for i, node in enumerate(node_list):
            if i % 100 == 0:
                print(f"Finding neighbors for node {node.id}/{len(node_list)}")
            distances, indices = self.kdtree.query(node.config, k=self.k_neighbors + 1)
            for idx in indices[1:]:  # skip self
                neighbor = node_list[idx]
                if neighbor.id > node.id:  # prevent double connections
                    all_pairs.append((node.id, neighbor.id))

        def try_connect(pair: t.Tuple[int, int]):
            i, j = pair
            node_i = self.roadmap[i]
            node_j = self.roadmap[j]

            if self.collision_checker.is_collision_free(node_i.config, node_j.config):
                cost = self._weighted_distance(node_i.config, node_j.config)
                return (i, j, cost)
            return None

        print(f"Checking {len(all_pairs)} potential edges in parallel...")
        with ThreadPoolExecutor() as executor:
            results = executor.map(try_connect, all_pairs)

        for idx,result in enumerate(results):
            if result is not None:
                i, j, cost = result
                if idx % 1000 == 0:
                    print(f"Adding edge {i}/{len(all_pairs)}")
                self.roadmap[i].add_edge(j, cost)
                self.roadmap[j].add_edge(i, cost)

    def save_roadmap(self, path: str):
        configs = np.array([node.config for node in self.roadmap.values()])
        edges = {node.id: node.edges for node in self.roadmap.values()}
        np.savez(path, configs=configs, edges=edges)

    def _add_node(self, config: np.ndarray) -> int:
        node = GraphNode(id=self.next_node_id, config=config)
        self.roadmap[self.next_node_id] = node
        self.next_node_id += 1
        return node.id

    def _weighted_distance(self, config_1: np.ndarray, config_2: np.ndarray) -> float:
        diff = config_1 - config_2
        return np.sqrt(np.sum(self.weights * diff**2))

    def add_box_obstacle(self, name, size, position, orientation=(0, 0, 0, 1), collision_margin=0.0):
        """
        Add a box obstacle to the planning scene with an optional collision margin.

        Args:
            name (str): Name of the obstacle.
            size (tuple): (x, y, z) dimensions of the actual box (meters).
            position (tuple): (x, y, z) center of the box (meters).
            orientation (tuple): Quaternion (x, y, z, w) orientation.
            collision_margin (float): Amount to inflate each dimension symmetrically (meters).
        """
        # Inflate size symmetrically
        inflated_size = tuple(s + 2 * collision_margin for s in size)

        box_pose = PoseStamped()
        box_pose.header.frame_id = self.robot_model.base_link
        box_pose.pose.position.x = position[0]
        box_pose.pose.position.y = position[1]
        box_pose.pose.position.z = position[2]
        box_pose.pose.orientation.x = orientation[0]
        box_pose.pose.orientation.y = orientation[1]
        box_pose.pose.orientation.z = orientation[2]
        box_pose.pose.orientation.w = orientation[3]

        self.scene.add_box(name, box_pose, size=inflated_size)
        rospy.sleep(1.0)
        rospy.loginfo(f"Added box '{name}' at {position} with inflated size {inflated_size} (original: {size}, margin: {collision_margin})")


if __name__ == "__main__":
    # Example usage
    robot_model = Robot()  # Replace with actual robot model initialization
    collision_checker = CollisionChecker()  # Replace with actual collision checker initialization

    #roadmap_name = "roadmap-restricted-collision-enabled-sampl_1000-k_5-step_01-w_1_1_08_05_01_01_01.npz"
    #roadmap_name = "roadmap-restricted-collision-enabled-sampl_1000-k_5-step_01-w_1_1_1_1_1_1_1.npz"
    roadmap_name = "roadmap-fixed-tool-with-obstacle-sampl_10000-k_10-step_01-w_1_1_1_1_1_1_1.npz"

    #weights = [1.0, 1.0, 0.8, 0.5, 0.1, 0.1, 0.1]  # Example weights for each joint
    weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]  # Default weights for all joints
    roadmap_builder = PRMRoadmapBuilder(robot_model = robot_model,
                                        collision_checker = collision_checker,
                                        n_nodes = 10000,
                                        k_neighbors = 10,
                                        weights = weights,
                                        path_to_save = roadmap_name,
                                        restrict_to_task_space=True,
                                        limit_tool_orientation=True)

    # Build the roadmap
    roadmap_builder.build_roadmap()
    
    # Save the roadmap to a file
    roadmap_path = os.path.join(os.path.dirname(__file__), "prm_roadmaps", roadmap_name)
    roadmap_builder.save_roadmap(roadmap_path)

    print(f"Roadmap saved to {roadmap_path}")
