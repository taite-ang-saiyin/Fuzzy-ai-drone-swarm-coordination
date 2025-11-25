from __future__ import annotations
import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass(order=True)
class Task:
    priority: int
    task_id: str = field(compare=False)
    target: np.ndarray = field(compare=False, default_factory=lambda: np.zeros(3))
    assigned_to: Optional[str] = field(compare=False, default=None)
    completed: bool = field(compare=False, default=False)

    def mark_done(self):
        self.completed = True
        self.assigned_to = None


class MissionPlanner:
    """
    Minimal task allocator that greedily assigns the nearest unassigned task
    to any drone that is currently idle. Tasks use a min-heap keyed by priority.
    """

    def __init__(self, tasks: List[Task] | None = None, assignment_radius: float = 2.0):
        self.assignment_radius = assignment_radius
        self._tasks: Dict[str, Task] = {}
        self._idle: Dict[str, bool] = {}
        self._queue: list[tuple[int, str]] = []

        for t in tasks or []:
            self.add_task(t)

    # -----------------------------
    # Task management
    # -----------------------------
    def add_task(self, task: Task):
        self._tasks[task.task_id] = task
        heapq.heappush(self._queue, (task.priority, task.task_id))

    def mark_idle(self, drone_id: str):
        self._idle[drone_id] = True

    def mark_busy(self, drone_id: str):
        self._idle[drone_id] = False

    # -----------------------------
    # Assignment logic
    # -----------------------------
    def assign(self, drone_positions: Dict[str, np.ndarray]):
        """
        Assign tasks to idle drones greedily by distance.
        """
        for drone_id, pos in drone_positions.items():
            if not self._idle.get(drone_id, True):
                continue

            best_task = self._pop_best_task(pos)
            if best_task:
                best_task.assigned_to = drone_id
                self._idle[drone_id] = False

    def current_goal(self, drone_id: str) -> Optional[np.ndarray]:
        for t in self._tasks.values():
            if t.assigned_to == drone_id and not t.completed:
                return t.target
        return None

    def update_progress(self, drone_id: str, position: np.ndarray):
        """
        Mark a task complete if the drone reached it.
        """
        for t in self._tasks.values():
            if t.assigned_to == drone_id and not t.completed:
                if np.linalg.norm(position - t.target) < self.assignment_radius:
                    t.mark_done()
                    self._idle[drone_id] = True
                break

    def pending_tasks(self) -> List[Task]:
        return [t for t in self._tasks.values() if not t.completed]

    # -----------------------------
    # Internal helpers
    # -----------------------------
    def _pop_best_task(self, drone_pos: np.ndarray) -> Optional[Task]:
        candidates: list[tuple[int, float, Task]] = []
        while self._queue:
            prio, tid = heapq.heappop(self._queue)
            task = self._tasks.get(tid)
            if task is None or task.completed or task.assigned_to:
                continue
            dist = np.linalg.norm(drone_pos - task.target)
            candidates.append((prio, dist, task))

        if not candidates:
            return None

        candidates.sort(key=lambda x: (x[0], x[1]))  # priority then distance
        chosen = candidates[0][2]

        # Rebuild queue without the chosen task
        for prio, _, task in candidates[1:]:
            heapq.heappush(self._queue, (prio, task.task_id))

        return chosen
