import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from agents.config import get_config
from agents.fuzzy_controller import create_fuzzy_controller
from agents import boids

# --- Config and Parameters ---
C = get_config()
NUM_DRONES = 6
NUM_OBSTACLES = 4
WINDOW_WIDTH = 60
WINDOW_HEIGHT = 30
SENSOR_RANGE = 6.0
DRONE_RADIUS = 0.4
OBSTACLE_RADIUS = 1.2
STEPS = 200
DT = 1.0

# --- World State ---
drones = np.zeros((NUM_DRONES, 2))
velocities = np.zeros((NUM_DRONES, 2))
stuck_histories = [[] for _ in range(NUM_DRONES)]
escape_steps = [0 for _ in range(NUM_DRONES)]
escape_modes = [None for _ in range(NUM_DRONES)]

drone_finished = [False for _ in range(NUM_DRONES)]

# --- Obstacles ---
obstacles = np.random.rand(NUM_OBSTACLES, 2) * [WINDOW_WIDTH-8, WINDOW_HEIGHT-8] + [4, 4]

# --- Fuzzy Controller ---
fuzzy = create_fuzzy_controller()

# --- V-formation Parameters ---
formation_spacing = 4.0
formation_angle = np.radians(30)

# --- Target Positions ---
START_POS = np.array([8, WINDOW_HEIGHT/2])
TARGET_POS = np.array([WINDOW_WIDTH-8, WINDOW_HEIGHT/2])

# --- Initialize drones in V-formation at START_POS ---
def v_formation_positions(center, num_drones, spacing, angle):
    positions = [center]
    for i in range(1, num_drones):
        side = -1 if i % 2 == 1 else 1
        rank = (i + 1) // 2
        dx = -rank * spacing * np.cos(angle)
        dy = side * rank * spacing * np.sin(angle)
        positions.append(center + np.array([dx, dy]))
    return np.array(positions)

drones[:,:] = v_formation_positions(START_POS, NUM_DRONES, formation_spacing, formation_angle)

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
    leader_pos = positions[leader_idx]
    leader_vel = velocities[leader_idx]
    n = idx
    if idx == leader_idx:
        return TARGET_POS
    side = -1 if n % 2 == 1 else 1
    rank = (n + 1) // 2
    heading = np.arctan2(TARGET_POS[1] - leader_pos[1], TARGET_POS[0] - leader_pos[0])
    dx = -rank * formation_spacing * np.cos(formation_angle)
    dy = side * rank * formation_spacing * np.sin(formation_angle)
    offset = np.array([dx, dy])
    return leader_pos + offset

def update(frame):
    plt.cla()
    # Draw obstacles
    for obs in obstacles:
        circle = plt.Circle(obs, OBSTACLE_RADIUS, color='gray', alpha=0.5)
        plt.gca().add_patch(circle)
    # Draw target
    plt.plot(TARGET_POS[0], TARGET_POS[1], 'kx', markersize=12)
    # Update and draw drones
    for i in range(NUM_DRONES):
        pos = drones[i]
        vel = velocities[i]
        if drone_finished[i]:
            velocities[i] = np.zeros_like(velocities[i])
            plt.plot(pos[0], pos[1], 'go')
            continue
        # Sensor directions: front, left, right, back
        base_angle = np.arctan2(TARGET_POS[1] - pos[1], TARGET_POS[0] - pos[0])
        sensor_dirs = [
            np.array([np.cos(base_angle), np.sin(base_angle)]),
            np.array([np.cos(base_angle + np.pi/2), np.sin(base_angle + np.pi/2)]),
            np.array([np.cos(base_angle - np.pi/2), np.sin(base_angle - np.pi/2)]),
            np.array([np.cos(base_angle + np.pi), np.sin(base_angle + np.pi)])
        ]
        sensor_dists = [get_sensor_distance(pos, d) for d in sensor_dirs]
        # Neighbor info
        neighbors = get_neighbors(i, drones, velocities)
        neighbor_distances = [np.linalg.norm(pos - p) for _, p, _, _ in neighbors] if neighbors else [WINDOW_WIDTH]
        nearest_neighbor_distance = min(neighbor_distances) if neighbor_distances else WINDOW_WIDTH
        # V-formation goal
        goal = v_formation_goal(i, drones, velocities, leader_idx=0)
        goal_vec = goal - pos
        distance_to_goal = np.linalg.norm(goal_vec)
        desired_heading = np.arctan2(goal_vec[1], goal_vec[0])
        current_heading = base_angle
        heading_error = (desired_heading - current_heading + np.pi) % (2 * np.pi) - np.pi
        # Stuck detection
        stuck_histories[i].append(np.copy(pos))
        if len(stuck_histories[i]) > 30:
            stuck_histories[i].pop(0)
        stuck_distance = np.linalg.norm(stuck_histories[i][-1] - stuck_histories[i][0]) if len(stuck_histories[i]) == 30 else 1.0
        stuck = stuck_distance < 0.1
        all_very_close = all(dist < 0.25 for dist in sensor_dists)
        # Fuzzy/boids hybrid blending
        fuzzy_out = fuzzy.get_advanced_avoidance_policy(sensor_dists, nearest_neighbor_distance, drone_id=f"drone{i}", stuck=stuck, all_very_close=all_very_close)
        fuzzy_thrust = fuzzy_out.get('thrust_adjustment', 0.0)
        fuzzy_yaw = fuzzy_out.get('yaw_rate', 0.0)
        # Boids logic
        boids_acc = boids.step(pos, vel, {f"drone{j}": {'pos': drones[j], 'vel': velocities[j]} for j in range(NUM_DRONES)}, goal=goal)
        # Hybrid blending (no RL in 2D)
        thrust_adjustment = fuzzy_thrust
        yaw_rate = fuzzy_yaw
        # Escape logic (persistent)
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
        # Combine boids, fuzzy, and goal seeking
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
        # Stop at target
        if np.linalg.norm(pos - TARGET_POS) < 1.0:
            drone_finished[i] = True
            velocities[i] = np.zeros_like(velocities[i])
            plt.plot(pos[0], pos[1], 'go')
            continue
        # Draw drone
        plt.plot(pos[0], pos[1], 'bo')
        # Draw sensor rays
        colors = ['r--', 'g--', 'b--', 'y--']
        for d, dist, c in zip(sensor_dirs, sensor_dists, colors):
            ray_end = pos + d * dist
            plt.plot([pos[0], ray_end[0]], [pos[1], ray_end[1]], c)
    plt.xlim(0, WINDOW_WIDTH)
    plt.ylim(0, WINDOW_HEIGHT)
    plt.title(f'Step {frame}')

if __name__ == '__main__':
    fig = plt.figure(figsize=(12,6))
    ani = FuncAnimation(fig, update, frames=STEPS, interval=100)
    plt.show()
