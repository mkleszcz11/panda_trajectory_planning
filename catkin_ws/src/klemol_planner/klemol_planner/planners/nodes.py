"""
Generic node definitions for use in motion planners.

Includes:
- TreeNode: for tree-based planners (e.g., RRT, RRT*)
- GraphNode: for graph-based planners (e.g., PRM)
"""

import typing as t
import numpy as np

class TreeNode:
    """
    Node used in tree-based planners like RRT or RRT*.

    Attributes:
        config: Joint configuration (np.ndarray).
        parent: Parent node (TreeNode or None).
        cost: Path cost from the root node.
        children: Optional list of children (used in RRT*).
    """
    def __init__(self, config: np.ndarray, parent: t.Optional['TreeNode'] = None):
        self.config = config
        self.parent = parent
        self.cost = float("inf") if parent else 0.0
        self.children: t.List['TreeNode'] = []

        if parent:
            parent.children.append(self)
            self.cost = parent.cost + TreeNode.compute_cost(config, parent.config)

    @staticmethod
    def compute_cost(config_a: np.ndarray, config_b: np.ndarray) -> float:
        """
        Compute the cost between two configurations.

        Args:
            config_a: First configuration (np.ndarray).
            config_b: Second configuration (np.ndarray).

        Returns:
            Cost (float) — here, Euclidean distance in joint space.

        Note:
            This can be replaced with a more complex cost function if needed.
            For instance,
             - we can penalize configurations that are close to obstacles.
             - we can include joint weights and penalise wrist more than elbow.
             - we can use path length in Cartesian space as a cost
             - ! we can use dynamic cost (velcoity, acceleration) !
        """
        return np.linalg.norm(config_a - config_b)


class GraphNode:
    """
    Node used in graph-based planners like PRM.

    Attributes:
        config: Joint configuration (np.ndarray).
        edges: Dict of neighbor nodes and their connection cost.
    """
    def __init__(self, config: np.ndarray):
        self.config = config
        self.edges: t.Dict['GraphNode', float] = {}

    def add_edge(self, other: 'GraphNode', cost: float):
        self.edges[other] = cost
        other.edges[self] = cost

    def remove_edge(self, other: 'GraphNode'):
        self.edges.pop(other, None)
        other.edges.pop(self, None)
