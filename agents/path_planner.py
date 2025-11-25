from __future__ import annotations
import heapq
import math
from typing import Iterable, List, Tuple
import numpy as np


class GridPathPlanner:
    """
    Coarse 2D A* planner on an occupancy grid. Obstacles are treated as disks
    and inflated with a safety margin so planned paths stay clear for the swarm.
    """

    def __init__(self, world_size: float = 50.0, resolution: float = 1.0, safety_margin: float = 1.0):
        self.world_size = world_size
        self.resolution = resolution
        self.safety_margin = safety_margin
        self.grid_size = int(math.ceil(world_size / resolution))
        self.occupancy = np.zeros((self.grid_size, self.grid_size), dtype=bool)

    # -----------------------------
    # Public API
    # -----------------------------
    def update_obstacles(self, obstacles: Iterable[np.ndarray], obstacle_radius: float):
        """Inflate obstacles into occupancy grid."""
        self.occupancy.fill(False)
        inflate = obstacle_radius + self.safety_margin
        for obs in obstacles:
            gx, gy = self._to_grid(obs[0]), self._to_grid(obs[1])
            radius_cells = int(math.ceil(inflate / self.resolution))
            for dx in range(-radius_cells, radius_cells + 1):
                for dy in range(-radius_cells, radius_cells + 1):
                    x = gx + dx
                    y = gy + dy
                    if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                        if math.hypot(dx, dy) * self.resolution <= inflate:
                            self.occupancy[x, y] = True

    def plan(self, start: np.ndarray, goal: np.ndarray) -> List[np.ndarray]:
        """
        Returns a list of 2D waypoints (world coordinates) from start to goal.
        Falls back to a straight line if no grid path is found.
        """
        start_node = (self._to_grid(start[0]), self._to_grid(start[1]))
        goal_node = (self._to_grid(goal[0]), self._to_grid(goal[1]))

        if not self._valid(start_node) or not self._valid(goal_node):
            return [goal]

        came_from = {}
        g_score = {start_node: 0}
        f_score = {start_node: self._heuristic(start_node, goal_node)}
        open_set = [(f_score[start_node], start_node)]

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal_node:
                return self._reconstruct_path(came_from, current)

            for neighbor in self._neighbors(current):
                tentative_g = g_score[current] + self.resolution
                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal_node)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        # Fallback: no path found
        return [goal]

    # -----------------------------
    # Internals
    # -----------------------------
    def _neighbors(self, node: Tuple[int, int]):
        x, y = node
        steps = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        for dx, dy in steps:
            nx, ny = x + dx, y + dy
            if self._valid((nx, ny)):
                yield (nx, ny)

    def _valid(self, node: Tuple[int, int]) -> bool:
        x, y = node
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size and not self.occupancy[x, y]

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _reconstruct_path(self, came_from, current) -> List[np.ndarray]:
        path = [self._to_world(current)]
        while current in came_from:
            current = came_from[current]
            path.append(self._to_world(current))
        path.reverse()
        return path

    def _to_grid(self, value: float) -> int:
        return int(np.clip(value / self.resolution, 0, self.grid_size - 1))

    def _to_world(self, node: Tuple[int, int]) -> np.ndarray:
        return np.array([(node[0] + 0.5) * self.resolution, (node[1] + 0.5) * self.resolution])
