from __future__ import annotations
import math
import numpy as np
from . import config as C

def _norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))

def _limit(v: np.ndarray, max_mag: float) -> np.ndarray:
    mag = _norm(v)
    if mag == 0:
        return v
    if mag > max_mag:
        return v * (max_mag / mag)
    return v

def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    # Returns angle in degrees between vectors a and b
    na, nb = _norm(a), _norm(b)
    if na == 0 or nb == 0:
        return 0.0
    cos_theta = np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)
    return math.degrees(math.acos(cos_theta))

def trimf(x: float, abc: list[float]) -> float:
    """Triangular membership function."""
    a, b, c = abc
    if x <= a or x >= c:
        return 0.0
    elif x < b:
        return (x - a) / (b - a)
    else:
        return (c - x) / (c - b)

def filter_neighbors(self_pos: np.ndarray,
                     self_vel: np.ndarray,
                     world_states: dict,
                     perception_radius: float,
                     fov_deg: float):
    neighbors = []
    for peer_id, s in world_states.items():
        p = np.array(s["pos"], dtype=float)
        v = np.array(s["vel"], dtype=float)
        d = p - self_pos
        dist = _norm(d)
        if dist == 0.0:
            continue
        dist_close = trimf(dist, [0, 0, C.SAFE_DISTANCE])
        dist_medium = trimf(dist, [C.SAFE_DISTANCE / 2, perception_radius / 2, perception_radius])
        if _norm(self_vel) > 1e-6:
            ang = _angle_between(self_vel, d)
            ang_in_fov = trimf(ang, [0, 0, fov_deg / 2])
        else:
            ang_in_fov = 1.0
        relevance = min(dist_medium, ang_in_fov) + 0.5 * dist_close
        relevance = np.clip(relevance, 0.0, 1.0)
        if relevance > 0.1:
            neighbors.append((peer_id, p, v, dist, relevance))
    return neighbors

def cohesion(self_pos: np.ndarray, neighbors) -> np.ndarray:
    if not neighbors:
        return np.zeros_like(self_pos)
    weights = np.array([rel for _, _, _, dist, rel in neighbors if dist > C.SAFE_DISTANCE])
    if len(weights) == 0:
        return np.zeros_like(self_pos)
    positions = np.array([p for _, p, _, dist, _ in neighbors if dist > C.SAFE_DISTANCE])
    center = np.average(positions, axis=0, weights=weights)
    desired = center - self_pos
    return _limit(desired, C.MAX_FORCE)

def alignment(self_vel: np.ndarray, neighbors) -> np.ndarray:
    if not neighbors:
        return np.zeros_like(self_vel)
    weights = np.array([rel for _, _, _, _, rel in neighbors])
    velocities = np.array([v for _, _, v, _, _ in neighbors])
    avg_vel = np.average(velocities, axis=0, weights=weights)
    steer = avg_vel - self_vel
    return _limit(steer, C.MAX_FORCE)

def separation(self_pos: np.ndarray, neighbors) -> np.ndarray:
    if not neighbors:
        return np.zeros_like(self_pos)
    force = np.zeros_like(self_pos)
    for _, p, _, dist, rel in neighbors:
        if dist < C.SAFE_DISTANCE:
            diff = self_pos - p
            diff_norm = _norm(diff)
            if diff_norm > 1e-6:
                sep_force = diff / (dist * dist)
                force += sep_force * rel * 2.0
    return _limit(force, C.MAX_FORCE)

def fuzzy_adjust_weights(self_pos: np.ndarray, neighbors, self_vel: np.ndarray):
    n_neigh = len(neighbors)
    density_low = trimf(n_neigh, [0, 0, 3])
    density_med = trimf(n_neigh, [2, 4, 6])
    density_high = trimf(n_neigh, [5, 10, 15])
    
    if neighbors:
        neigh_vels = np.array([v for _, _, v, _, _ in neighbors])
        vel_var = np.var(neigh_vels, axis=0).mean()
    else:
        vel_var = 0
    var_coherent = trimf(vel_var, [0, 0, 1.0])
    var_scattered = trimf(vel_var, [2.0, 5.0, C.MAX_SPEED**2])
    
    if neighbors:
        positions = np.array([p for _, p, _, _, _ in neighbors])
        center = np.mean(positions, axis=0)
        dist_to_center = _norm(self_pos - center)
    else:
        dist_to_center = 0
    dist_far = trimf(dist_to_center, [C.PERCEPTION_RADIUS / 2, C.PERCEPTION_RADIUS, C.PERCEPTION_RADIUS * 1.5])
    
    rule1_sep = min(density_high, var_scattered)
    rule1_coh = min(density_high, var_scattered)
    rule2_sep = min(density_low, var_coherent)
    rule2_coh = min(density_low, var_coherent)
    rule3_coh = dist_far
    rule4_align = var_scattered
    
    sep_weight = (rule1_sep * 3.5 + rule2_sep * 1.0 + density_med * 2.0) / max(1e-6, rule1_sep + rule2_sep + density_med)
    coh_weight = (rule1_coh * 0.5 + rule2_coh * 1.5 + rule3_coh * 2.0) / max(1e-6, rule1_coh + rule2_coh + rule3_coh)
    align_weight = (rule4_align * 2.0 + var_coherent * 0.8) / max(1e-6, rule4_align + var_coherent)
    
    return {
        'cohesion': C.WEIGHT_COHESION * coh_weight,
        'alignment': C.WEIGHT_ALIGNMENT * align_weight,
        'separation': C.WEIGHT_SEPARATION * sep_weight,
    }

def goal_seek(self_pos: np.ndarray, goal: np.ndarray | None) -> np.ndarray:
    if goal is None:
        return np.zeros_like(self_pos)
    desired = goal - self_pos
    return _limit(desired, C.MAX_FORCE)

def step(self_pos: np.ndarray,
         self_vel: np.ndarray,
         world_states: dict,
         goal: np.ndarray | None = None) -> np.ndarray:
    # Get neighbors within perception & FOV
    neighbors = filter_neighbors(self_pos, self_vel, world_states,
                                 C.PERCEPTION_RADIUS, C.FOV_DEG)
    
    # Use fuzzy weight adjustment for dynamic behavior
    weights = fuzzy_adjust_weights(self_pos, neighbors, self_vel)
    
    # Compute steering components with fuzzy-adjusted weights
    steer_c = weights['cohesion'] * cohesion(self_pos, neighbors)
    steer_a = weights['alignment'] * alignment(self_vel, neighbors)
    steer_s = weights['separation'] * separation(self_pos, neighbors)
    steer_g = C.WEIGHT_GOAL * goal_seek(self_pos, goal)

    steer = steer_c + steer_a + steer_s + steer_g
    
    # Fallback: small random force if no neighbors
    if len(neighbors) == 0:
        steer += np.random.uniform(-0.1, 0.1, 3) * C.MAX_FORCE
    
    return _limit(steer, C.MAX_FORCE)

def integrate(pos: np.ndarray, vel: np.ndarray, acc: np.ndarray, dt: float):
    vel = vel + acc * dt
    vel = _limit(vel, C.MAX_SPEED)
    pos = pos + vel * dt
    return pos, vel
