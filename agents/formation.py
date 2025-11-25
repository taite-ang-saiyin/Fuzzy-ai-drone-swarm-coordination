import numpy as np


class FormationController:
    """
    Lightweight formation helper that turns an index -> offset into an
    absolute desired position based on the leader pose. Offsets are expressed
    in the leader's body frame (x forward, y left).
    """

    def __init__(self, formation: str = "v", spacing: float = 4.0, altitude: float | None = None):
        self.formation = formation
        self.spacing = spacing
        self.altitude = altitude

    def set_formation(self, formation: str):
        self.formation = formation

    def set_spacing(self, spacing: float):
        self.spacing = spacing

    # -----------------------------
    # Public helpers
    # -----------------------------
    def desired_position(self, leader_pos: np.ndarray, leader_heading: float, drone_index: int) -> np.ndarray:
        """
        Return the absolute desired position for a given follower index.
        Index 0 is the leader and returns the leader position unchanged.
        """
        if drone_index == 0:
            return leader_pos

        offset = self._offset_body_frame(drone_index)
        # Rotate offset into world frame using leader heading (yaw)
        cos_h = np.cos(leader_heading)
        sin_h = np.sin(leader_heading)
        rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
        world_xy = leader_pos[:2] + rot @ offset[:2]

        z = leader_pos[2] if self.altitude is None else self.altitude
        return np.array([world_xy[0], world_xy[1], z])

    # -----------------------------
    # Internal formation shapes
    # -----------------------------
    def _offset_body_frame(self, drone_index: int) -> np.ndarray:
        """Return offset in the leader body frame."""
        spacing = self.spacing
        # Default line behind leader for safety
        if self.formation == "line":
            return np.array([-drone_index * spacing, 0.0, 0.0])

        if self.formation == "column":
            # Stagger laterally every other drone
            side = -1 if drone_index % 2 else 1
            rank = (drone_index + 1) // 2
            return np.array([-rank * spacing, side * 0.4 * spacing, 0.0])

        if self.formation == "diamond":
            pattern = [
                np.array([0.0, 0.0, 0.0]),  # leader slot (index 0)
                np.array([-spacing, spacing * 0.6, 0.0]),
                np.array([-spacing, -spacing * 0.6, 0.0]),
                np.array([-spacing * 2.0, 0.0, 0.0]),
                np.array([-spacing * 3.0, 0.0, 0.0]),
            ]
            if drone_index < len(pattern):
                return pattern[drone_index]
            # Fall back to a line behind the diamond
            return np.array([-(drone_index) * spacing, 0.0, 0.0])

        if self.formation == "circle":
            # Place followers on a circle around the leader
            angle = 2 * np.pi * (drone_index - 1) / max(1, drone_index)
            return np.array([np.cos(angle) * spacing, np.sin(angle) * spacing, 0.0])

        # Default and "v" formation (leader at index 0)
        side = -1 if drone_index % 2 == 1 else 1  # alternate sides
        rank = (drone_index + 1) // 2
        dx = -rank * spacing
        dy = side * rank * spacing * 0.6
        return np.array([dx, dy, 0.0])
