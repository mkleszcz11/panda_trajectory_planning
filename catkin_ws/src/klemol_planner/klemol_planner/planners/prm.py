from dataclasses import dataclass, field
import numpy as np
import typing as t
import heapq
from scipy.spatial import cKDTree

from klemol_planner.planners.base import Planner
from klemol_planner.environment.robot_model import Robot
from klemol_planner.environment.collision_checker import CollisionChecker
from klemol_planner.goals.point_with_orientation import PointWithOrientation

from klemol_planner.planners.nodes import GraphNode
from trac_ik_python.trac_ik import IK

import os
from concurrent.futures import ThreadPoolExecutor

class PRMPlanner(Planner):
    def __init__(self,
                 robot_model: Robot,
                 collision_checker: CollisionChecker,
                 parameters: dict):
        super().__init__(robot_model, collision_checker, parameters)
        self.n_nodes: int = parameters.get("n_nodes", 300)
        self.k_neighbors: int = parameters.get("k_neighbors", 10)
        self.step_size: float = parameters.get("step_size", 0.1)
        self.weights: np.ndarray = np.array(parameters.get("weights", [1.0]*7))
        self.roadmap: t.Dict[int, GraphNode] = {}
        self.kdtree: t.Optional[cKDTree] = None
        self.next_node_id: int = 0
        self.goal_configs: t.List[np.ndarray] = []

    def plan(self) -> t.Tuple[t.List[np.ndarray], bool]:
        """
        Builds roadmap, connects start/goal, runs A*
        """
        if self.start_config is None or self.goal_pose is None:
            raise ValueError("Start and goal must be set before planning.")

        # Look for the file in config directory. If not found, build a new roadmap and save it.
        config_path = os.path.join(os.path.dirname(__file__), "../../config/prm_roadmap.npz")
        if not self.roadmap:
            roadmap_loaded = self._load_roadmap(config_path)
            if not roadmap_loaded:
                self._build_roadmap()
                self._save_roadmap(config_path)


        self.goal_configs = self._generate_goal_configurations(self.goal_pose)
        if not self.goal_configs:
            return [], False

        print(f"--- connecting start and goal ---")
        start_id = self._add_query_node(self.start_config)
        goal_ids = [self._add_query_node(cfg) for cfg in self.goal_configs]
        #self.connect_start_and_goal_nodes_to_the_grid(start_id, goal_ids)
        print(f"running A*")
        return self._a_star(start_id, goal_ids)

    def _build_roadmap(self) -> None:
        """
        Samples valid nodes, connects neighbors using KDTree + collision check
        """
        print(f"--- building roadmap ---")
        self.roadmap.clear()
        valid_configs = []

        i = 0
        while len(valid_configs) < self.n_nodes:
            if i % 1000 == 0:
                print(f"Sampling node {len(valid_configs)}/{self.n_nodes}")
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
            if i % 1000 == 0:
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

            # if self._is_collision_free_path(node_i.config, node_j.config):
            cost = self._weighted_distance(node_i.config, node_j.config)
            return (i, j, cost)
            # return None

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

    def _load_roadmap(self, path: str):
        """
        Loads roadmap from file, if it exists.

        Returns True if loaded successfully, False otherwise.
        """
        print(f"Loading roadmap from {path}")
        try:
            data = np.load(path, allow_pickle=True)
            configs = data["configs"]
            edges = data["edges"].item()  # because it's an object

            self.roadmap.clear()
            self.next_node_id = 0
            for i, config in enumerate(configs):
                node = GraphNode(id=i, config=config)
                node.edges = edges.get(i, {})
                self.roadmap[i] = node
                self.next_node_id = max(self.next_node_id, i + 1)

            self.kdtree = cKDTree(configs)
            return True
        except FileNotFoundError:
            print(f"Roadmap file not found at {path}. Building a new roadmap.")
            return False

    def _save_roadmap(self, path: str):
        configs = np.array([node.config for node in self.roadmap.values()])
        edges = {node.id: node.edges for node in self.roadmap.values()}
        np.savez(path, configs=configs, edges=edges)

    def _add_node(self, config: np.ndarray) -> int:
        node = GraphNode(id=self.next_node_id, config=config)
        self.roadmap[self.next_node_id] = node
        self.next_node_id += 1
        return node.id

    def _add_query_node(self, config: np.ndarray) -> int:
        node_id = self._add_node(config)
        node = self.roadmap[node_id]
        distances, indices = self.kdtree.query(config, k=self.k_neighbors)
        for idx in indices:
            neighbor = list(self.roadmap.values())[idx]
            if self._is_collision_free_path(config, neighbor.config):
                cost = self._weighted_distance(config, neighbor.config)
                node.add_edge(neighbor.id, cost)
                neighbor.add_edge(node.id, cost)
        return node_id

    def _a_star(self, start_id: int, goal_ids: t.List[int]) -> t.Tuple[t.List[np.ndarray], bool]:
        """
        Graph search using A*, stops on first goal match
        """
        open_set = [(0.0, start_id)]
        came_from: t.Dict[int, int] = {}
        g_score = {start_id: 0.0}
        f_score = {start_id: self._heuristic(self.roadmap[start_id].config)}

        closed_set = set()

        while open_set:
            _, current = heapq.heappop(open_set)

            if current in goal_ids:
                return self._reconstruct_path(current, came_from), True

            closed_set.add(current)

            for neighbor_id, cost in self.roadmap[current].edges.items():
                if neighbor_id in closed_set:
                    continue

                tentative_g = g_score[current] + cost
                if tentative_g < g_score.get(neighbor_id, float('inf')):
                    came_from[neighbor_id] = current
                    g_score[neighbor_id] = tentative_g
                    f = tentative_g + self._heuristic(self.roadmap[neighbor_id].config)
                    f_score[neighbor_id] = f
                    heapq.heappush(open_set, (f, neighbor_id))

        return [], False

    def _reconstruct_path(self, goal_id: int, came_from: t.Dict[int, int]) -> t.List[np.ndarray]:
        path = []
        current = goal_id
        while current in came_from:
            path.append(self.roadmap[current].config)
            current = came_from[current]
        path.append(self.roadmap[current].config)
        return path[::-1]

    def _is_collision_free_path(self, from_q: np.ndarray, to_q: np.ndarray) -> bool:
        dist = np.linalg.norm(to_q - from_q)
        steps = max(2, int(np.ceil(dist / self.step_size)))
        for i in range(1, steps):
            alpha = i / steps
            interp = (1 - alpha) * from_q + alpha * to_q
            if not self.robot_model.is_within_limits(interp):
                return False
            # if self.collision_checker.is_in_collision(interp):
            #     return False
        return True

    def _weighted_distance(self, config_1: np.ndarray, config_2: np.ndarray) -> float:
        diff = config_1 - config_2
        return np.sqrt(np.sum(self.weights * diff**2))

    def _heuristic(self, config: np.ndarray) -> float:
        return min(self._weighted_distance(config, g) for g in self.goal_configs)

    def _generate_goal_configurations(self, goal_pose: PointWithOrientation) -> t.List[np.ndarray]:
        goal_configs = []
        # ik_solver = self.robot_model.get_custom_ik_solver(timeout=0.1)
        ik_solver = IK(
            base_link=self.robot_model.base_link,
            tip_link=self.robot_model.ee_link,
            urdf_string=self.robot_model.urdf_string,
            timeout=0.1,
            solve_type="Speed",
        )
        attempts = 0
        while len(goal_configs) < 5 and attempts < 50:
            seed = self.robot_model.sample_random_configuration()
            q = self.robot_model.ik_with_custom_solver(goal_pose, solver=ik_solver, seed=seed)
            attempts += 1
            if q is None:
                continue
            if not self.robot_model.is_within_limits(q) or self.collision_checker.is_in_collision(q):
                continue
            if any(np.linalg.norm(q - existing) < 1e-2 for existing in goal_configs):
                continue
            goal_configs.append(q)
        return goal_configs