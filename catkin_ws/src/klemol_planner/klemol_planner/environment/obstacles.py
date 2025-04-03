
# NOT USED FOR NOW AS WE ARE RELYING ON MOVEIT FOR SCENE DESCRIPTION

import typing as t
import numpy as np

class BoxObstacle:
    """
    Axis-Aligned Bounding Box (AABB) obstacle.

    Attributes:
        min_corner: np.ndarray of shape (3,) representing the minimum x, y, z.
        max_corner: np.ndarray of shape (3,) representing the maximum x, y, z.
    """
    def __init__(self, min_corner: t.List[float], max_corner: t.List[float]):
        self.min_corner = np.array(min_corner)
        self.max_corner = np.array(max_corner)

    def is_point_inside(self, point: np.ndarray) -> bool:
        """
        Check if a point lies inside the box.

        Args:
            point: 3D point as np.ndarray

        Return:
            True if inside, False otherwise.
        """
        return np.all(point >= self.min_corner) and np.all(point <= self.max_corner)


class SphereObstacle:
    """
    Spherical obstacle.

    Attributes:
        center: np.ndarray of shape (3,) representing the center of the sphere.
        radius: Radius of the sphere.
    """
    def __init__(self, center: t.List[float], radius: float):
        self.center = np.array(center)
        self.radius = radius

    def is_point_inside(self, point: np.ndarray) -> bool:
        """
        Check if a point lies inside the sphere.

        Args:
            point: 3D point as np.ndarray

        Return:
            True if inside, False otherwise.
        """
        return np.linalg.norm(point - self.center) <= self.radius


class ObstacleManager:
    """
    Stores and manages multiple obstacles (boxes and spheres).
    """
    def __init__(self):
        self.boxes: t.List[BoxObstacle] = []
        self.spheres: t.List[SphereObstacle] = []

    def add_box(self, min_corner: t.List[float], max_corner: t.List[float]) -> None:
        self.boxes.append(BoxObstacle(min_corner, max_corner))

    def add_sphere(self, center: t.List[float], radius: float) -> None:
        self.spheres.append(SphereObstacle(center, radius))

    def get_all_obstacles(self) -> t.Tuple[t.List[BoxObstacle], t.List[SphereObstacle]]:
        return self.boxes, self.spheres
