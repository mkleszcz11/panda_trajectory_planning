import numpy as np
import typing as t

from klemol_planner.environment.robot_model import RobotModel
from klemol_planner.environment.collision_checker import CollisionChecker


class PathShortcutter:
    """
    Class to shortcut a path.
    """
    def __init__(self, collision_checker):
        self.collision_checker = collision_checker

    def interpolate(self, config_a: np.ndarray, config_b: np.ndarray, num_points: int = 10) -> t.List[np.ndarray]:
        """
        Linearly interpolate between two configurations.

        Args:
            config_a: Start joint configuration.
            config_b: End joint configuration.
            num_points: Number of intermediate samples.

        Returns:
            List of interpolated joint configurations.
        """
        return [config_a + (config_b - config_a) * float(i) / (num_points - 1) for i in range(num_points)]

    def is_collision_free(self, start: np.ndarray, end: np.ndarray, num_samples: int = 10) -> bool:
        """
        Check if straight-line interpolation between two joint configurations is collision-free.

        Args:
            start: Start configuration.
            end: End configuration.
            num_samples: Number of interpolation points.

        Returns:
            True if all interpolated points are collision free.
        """
        for point in self.interpolate(start, end, num_samples):
            if self.collision_checker.is_in_collision(point):
                return False
        return True

    def generate_a_shortcutted_path(self, path: t.List[np.ndarray]) -> t.List[np.ndarray]:
        """
        Analyze a path and shortcut it as much as possible using straight-line segments.

        The shortcutting logic checks if intermediate points can be skipped without collision.

        Args:
            path: List of joint configurations representing the original path.

        Returns:
            List of joint configurations forming a shorter, still collision-free path.
        """
        if not path:
            return []
        print("WOLOLOLOLOLOLOLOLOLOLOLOLOLOLOLOLOLOLOLOLOLOLOLOLOLO")
        print(f"Original path length: {len(path)}")
        i = 0
        new_path = [path[0]]

        while i < len(path) - 1:
            found = False
            for j in range(len(path) - 1, i, -1):
                if self.is_collision_free(path[i], path[j]):
                    new_path.append(path[j])
                    i = j
                    found = True
                    break
            if not found:
                new_path.append(path[i + 1])
                i += 1
        print(f"Shortcutted path length: {len(new_path)}")
        print("WOLOLOLOLOLOLOLOLOLOLOLOLOLOLOLOLOLOLOLOLOLOLOLOLOLO")
        return new_path
