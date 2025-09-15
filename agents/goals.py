import numpy as np

class WaypointManager:
    def __init__(self, global_waypoints=None, per_drone_waypoints=None):
        # Global waypoints (all drones share)
        self.global_waypoints = global_waypoints or [np.array([5.0, 0.0, 1.0]), np.array([-5.0, 0.0, 1.0])]
        self.global_idx = 0

        # Per-drone waypoints: {drone_id: [waypoint1, waypoint2, ...]}
        self.per_drone_waypoints = per_drone_waypoints or {}
        self.per_drone_idx = {drone_id: 0 for drone_id in self.per_drone_waypoints}

    def get_global_goal(self):
        if self.global_idx >= len(self.global_waypoints):
            return self.global_waypoints[-1]
        return self.global_waypoints[self.global_idx]

    def global_goal_reached(self, current_pos, thresh=0.8):
        goal = self.get_global_goal()
        if np.linalg.norm(current_pos - goal) < thresh:
            self.global_idx = min(self.global_idx + 1, len(self.global_waypoints)-1)
            return True
        return False

    def get_drone_goal(self, drone_id):
        waypoints = self.per_drone_waypoints.get(drone_id)
        idx = self.per_drone_idx.get(drone_id, 0)
        if not waypoints:
            return None
        if idx >= len(waypoints):
            return waypoints[-1]
        return waypoints[idx]

    def drone_goal_reached(self, drone_id, current_pos, thresh=0.8):
        goal = self.get_drone_goal(drone_id)
        if goal is None:
            return False
        if np.linalg.norm(current_pos - goal) < thresh:
            idx = self.per_drone_idx.get(drone_id, 0)
            self.per_drone_idx[drone_id] = min(idx + 1, len(self.per_drone_waypoints[drone_id])-1)
            return True
        return False

    def set_global_waypoints(self, waypoints):
        self.global_waypoints = waypoints
        self.global_idx = 0

    def set_drone_waypoints(self, drone_id, waypoints):
        self.per_drone_waypoints[drone_id] = waypoints
        self.per_drone_idx[drone_id] = 0

    def reset(self):
        self.global_idx = 0
        self.per_drone_idx = {drone_id: 0 for drone_id in self.per_drone_waypoints}
