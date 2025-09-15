import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from agents.config import get_config
from agents.fuzzy_controller import create_fuzzy_controller
from agents import boids
from agents.goals import WaypointManager

# --- Config and Parameters ---
C = get_config()
NUM_DRONES = 6
NUM_OBSTACLES = 4
WORLD_SIZE = 30
SENSOR_RANGE = 6.0
DRONE_RADIUS = 0.4
OBSTACLE_RADIUS = 1.2
STEPS = 200
DT = 1.0

# --- World State ---
drones = np.random.rand(NUM_DRONES, 2) * (WORLD_SIZE - 2) + 1
velocities = np.zeros((NUM_DRONES, 2))
stuck_histories = [[] for _ in range(NUM_DRONES)]
escape_steps = [0 for _ in range(NUM_DRONES)]
escape_modes = [None for _ in range(NUM_DRONES)]

# --- Obstacles ---
obstacles = np.random.rand(NUM_OBSTACLES, 2) * (WORLD_SIZE - 4) + 2

# Per-drone goal indices for multi-waypoint navigation
per_drone_goal_idx = [0 for _ in range(NUM_DRONES)]
# Pause timers for each drone at each waypoint
pause_timers = [0 for _ in range(NUM_DRONES)]
PAUSE_DURATION = 15  # frames to pause at each waypoint
# Track if each drone is finished
drone_finished = [False for _ in range(NUM_DRONES)]

# --- Fuzzy Controller ---
fuzzy = create_fuzzy_controller()

# --- Waypoint Manager (global for all drones) ---
goal_manager = WaypointManager(global_waypoints=[
    np.array([WORLD_SIZE-2, 2, 0]),
    np.array([2, WORLD_SIZE-2, 0]),
    np.array([WORLD_SIZE/2, WORLD_SIZE/2, 0])
])

# --- Helper Functions ---
def get_sensor_distance(drone_pos, direction):
    direction = direction / (np.linalg.norm(direction) + 1e-6)
    min_dist = SENSOR_RANGE
    for obs in obstacles:
        rel = obs - drone_pos
        proj = np.dot(rel, direction)
        if 0 < proj < SENSOR_RANGE:
            closest = drone_pos + proj * direction
            dist_to_obs = np.linalg.norm(obs - closest)
            if dist_to_obs < OBSTACLE_RADIUS + DRONE_RADIUS:
                min_dist = min(min_dist, proj)
    return min_dist

def get_neighbors(idx, positions, velocities):
    self_pos = positions[idx]
    self_vel = velocities[idx]
    neighbors = []
    for j, (p, v) in enumerate(zip(positions, velocities)):
        if j == idx:
            continue
        d = p - self_pos
        dist = np.linalg.norm(d)
        if dist == 0.0 or dist > C['boids']['perception_radius']:
            continue
        if np.linalg.norm(self_vel) > 1e-6:
            ang = np.degrees(np.arccos(np.clip(np.dot(self_vel, d) / ((np.linalg.norm(self_vel)+1e-6) * (np.linalg.norm(d)+1e-6)), -1.0, 1.0)))
            if ang > (C['boids']['fov_deg'] / 2.0):
                continue
        neighbors.append((j, p, v, dist))
    return neighbors

def v_formation_goal(idx, positions, velocities, leader_idx=0):
    # Arrow (V) formation logic, 2D version
    leader_pos = positions[leader_idx]
    leader_vel = velocities[leader_idx]
    n = idx
    # Each drone uses its own goal waypoint (cycling through global waypoints)
    if idx == leader_idx:
        return goal_manager.global_waypoints[per_drone_goal_idx[idx]][:2]
    formation_spacing = 6.0
    formation_angle = np.radians(30)
    side = -1 if n % 2 == 1 else 1
    rank = (n + 1) // 2
    heading = np.arctan2(leader_vel[1], leader_vel[0]) if np.linalg.norm(leader_vel) > 1e-3 else 0.0
    dx = np.cos(heading) * (-rank * formation_spacing * np.cos(formation_angle))
    dy = np.sin(heading) * (-rank * formation_spacing * np.cos(formation_angle))
    perp_angle = heading + side * formation_angle
    px = np.cos(perp_angle) * (rank * formation_spacing * np.sin(formation_angle))
    py = np.sin(perp_angle) * (rank * formation_spacing * np.sin(formation_angle))
    offset = np.array([dx + px, dy + py])
    return leader_pos + offset

def update(frame):
    plt.cla()
    # Draw obstacles
    for obs in obstacles:
        circle = plt.Circle(obs, OBSTACLE_RADIUS, color='gray', alpha=0.5)
        plt.gca().add_patch(circle)
    # Update and draw drones
    for i in range(NUM_DRONES):
        pos = drones[i]
        vel = velocities[i]
        # If drone is finished, keep it stopped and green
        if drone_finished[i]:
            velocities[i] = np.zeros_like(velocities[i])
            plt.plot(pos[0], pos[1], 'go')  # green = finished
            # Draw goal for reference
            waypoints = goal_manager.global_waypoints
            goal_idx = per_drone_goal_idx[i]
            goal = waypoints[goal_idx][:2]
            plt.plot(goal[0], goal[1], 'kx')
            continue
        # --- Sensor directions: front, left, right, back ---
        base_angle = np.arctan2(vel[1], vel[0]) if np.linalg.norm(vel) > 0 else 0
        sensor_dirs = [
            np.array([np.cos(base_angle), np.sin(base_angle)]),
            np.array([np.cos(base_angle + np.pi/2), np.sin(base_angle + np.pi/2)]),
            np.array([np.cos(base_angle - np.pi/2), np.sin(base_angle - np.pi/2)]),
            np.array([np.cos(base_angle + np.pi), np.sin(base_angle + np.pi)])
        ]
        sensor_dists = [get_sensor_distance(pos, d) for d in sensor_dirs]
        # --- Neighbor info ---
        neighbors = get_neighbors(i, drones, velocities)
        neighbor_distances = [np.linalg.norm(pos - p) for _, p, _, _ in neighbors] if neighbors else [WORLD_SIZE]
        nearest_neighbor_distance = min(neighbor_distances) if neighbor_distances else WORLD_SIZE
        # --- Formation goal ---
        goal = v_formation_goal(i, drones, velocities, leader_idx=0)
        goal_vec = goal - pos
        distance_to_goal = np.linalg.norm(goal_vec)
        desired_heading = np.arctan2(goal_vec[1], goal_vec[0])
        current_heading = base_angle
        heading_error = (desired_heading - current_heading + np.pi) % (2 * np.pi) - np.pi
        # --- Stuck detection ---
        stuck_histories[i].append(np.copy(pos))
        if len(stuck_histories[i]) > 30:
            stuck_histories[i].pop(0)
        stuck_distance = np.linalg.norm(stuck_histories[i][-1] - stuck_histories[i][0]) if len(stuck_histories[i]) == 30 else 1.0
        stuck = stuck_distance < 0.1
        all_very_close = all(dist < 0.25 for dist in sensor_dists)
        # --- Fuzzy/boids hybrid blending ---
        fuzzy_out = fuzzy.get_advanced_avoidance_policy(sensor_dists, nearest_neighbor_distance, drone_id=f"drone{i}", stuck=stuck, all_very_close=all_very_close)
        fuzzy_thrust = fuzzy_out.get('thrust_adjustment', 0.0)
        fuzzy_yaw = fuzzy_out.get('yaw_rate', 0.0)
        # --- Boids logic ---
        boids_acc = boids.step(pos, vel, {f"drone{j}": {'pos': drones[j], 'vel': velocities[j]} for j in range(NUM_DRONES)}, goal=goal)
        # --- Hybrid blending (no RL in 2D) ---
        thrust_adjustment = fuzzy_thrust
        yaw_rate = fuzzy_yaw
        # --- Escape logic (persistent) ---
        if stuck and all_very_close:
            if escape_steps[i] == 0:
                escape_modes[i] = np.random.choice(['reverse', 'turn', 'lateral'])
                escape_steps[i] = 15
            if escape_modes[i] == 'reverse':
                thrust_adjustment = -1.0
                yaw_rate = 0.0
            elif escape_modes[i] == 'turn':
                thrust_adjustment = 0.0
                yaw_rate = np.random.choice([-2.0, 2.0])
            elif escape_modes[i] == 'lateral':
                thrust_adjustment = 0.0
                yaw_rate = np.random.choice([-1.5, 1.5])
            escape_steps[i] -= 1
            if escape_steps[i] <= 0:
                escape_modes[i] = None
                escape_steps[i] = 0
        else:
            escape_modes[i] = None
            escape_steps[i] = 0
        # --- Combine boids, fuzzy, and goal seeking ---
        # Use yaw_rate to rotate movement direction, thrust to scale speed
        angle = base_angle + yaw_rate * 0.12 + heading_error * 0.2
        move_vec = np.array([np.cos(angle), np.sin(angle)])
        w_fuzzy = max(0.5, 1.0 - sensor_dists[0] / SENSOR_RANGE)
        w_boids = 1.0 - w_fuzzy
        combined = w_fuzzy * move_vec + w_boids * boids_acc[:2]
        if np.linalg.norm(combined) > 0:
            combined = combined / (np.linalg.norm(combined) + 1e-6)
        speed = 0.22 + thrust_adjustment * 0.12
        velocities[i] = combined * speed
        drones[i] += velocities[i]
        # --- Waypoint update logic ---
        if np.linalg.norm(pos - goal) < 1.0:
            # If at last waypoint, mark as finished
            if per_drone_goal_idx[i] == len(goal_manager.global_waypoints) - 1:
                drone_finished[i] = True
                velocities[i] = np.zeros_like(velocities[i])
                plt.plot(pos[0], pos[1], 'go')
                plt.plot(goal[0], goal[1], 'kx')
                continue
            else:
                per_drone_goal_idx[i] += 1
                goal = goal_manager.global_waypoints[per_drone_goal_idx[i]][:2]
                goal_vec = goal - pos
                distance_to_goal = np.linalg.norm(goal_vec)
                desired_heading = np.arctan2(goal_vec[1], goal_vec[0])
                heading_error = (desired_heading - base_angle + np.pi) % (2 * np.pi) - np.pi
        # --- Draw drone ---
        plt.plot(pos[0], pos[1], 'bo')
        # --- Draw sensor rays ---
        colors = ['r--', 'g--', 'b--', 'y--']
        for d, dist, c in zip(sensor_dirs, sensor_dists, colors):
            ray_end = pos + d * dist
            plt.plot([pos[0], ray_end[0]], [pos[1], ray_end[1]], c)
        # --- Draw goal ---
        plt.plot(goal[0], goal[1], 'kx')
    plt.xlim(0, WORLD_SIZE)
    plt.ylim(0, WORLD_SIZE)
    plt.title(f'Step {frame}')

if __name__ == '__main__':
    fig = plt.figure(figsize=(8,8))
    ani = FuncAnimation(fig, update, frames=STEPS, interval=100)
    plt.show()
