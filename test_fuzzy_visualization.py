import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from agents.fuzzy_controller import create_fuzzy_controller
from agents import boids



# Parameters
NUM_DRONES = 5
NUM_OBSTACLES = 3
WORLD_SIZE = 20
SENSOR_RANGE = 5.0
DRONE_RADIUS = 0.3
OBSTACLE_RADIUS = 1.0
STEPS = 100
DT = 1.0
COMM_RANGE = 10.0  # Communication range for boids logic

# Initialize drones (random positions, 2D)
drones = np.random.rand(NUM_DRONES, 2) * (WORLD_SIZE - 2) + 1
velocities = np.zeros((NUM_DRONES, 2))

# Initialize obstacles (random positions)
obstacles = np.random.rand(NUM_OBSTACLES, 2) * (WORLD_SIZE - 4) + 2

# Fuzzy controller
fuzzy = create_fuzzy_controller()

# Boids config (2D version)
PERCEPTION_RADIUS = 8.0
FOV_DEG = 270.0
SAFE_DISTANCE = 2.0
WEIGHT_COHESION = 1.0
WEIGHT_ALIGNMENT = 0.6
WEIGHT_SEPARATION = 1.8
WEIGHT_GOAL = 0.3

# Dummy goal (center)
GOAL = np.array([WORLD_SIZE/2, WORLD_SIZE/2])



# Helper: compute distance to nearest obstacle in a given direction
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


# Helper: get neighbors for boids (2D) with communication range
def get_neighbors(idx, positions, velocities):
    self_pos = positions[idx]
    self_vel = velocities[idx]
    neighbors = []
    for j, (p, v) in enumerate(zip(positions, velocities)):
        if j == idx:
            continue
        d = p - self_pos
        dist = np.linalg.norm(d)
        if dist == 0.0 or dist > PERCEPTION_RADIUS or dist > COMM_RANGE:
            continue
        if np.linalg.norm(self_vel) > 1e-6:
            ang = np.degrees(np.arccos(np.clip(np.dot(self_vel, d) / ((np.linalg.norm(self_vel)+1e-6) * (np.linalg.norm(d)+1e-6)), -1.0, 1.0)))
            if ang > (FOV_DEG / 2.0):
                continue
        neighbors.append((j, p, v, dist))
    return neighbors



# Animation update function
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
        # Sensor directions: front, left, right, back
        if np.linalg.norm(vel) > 0:
            base_angle = np.arctan2(vel[1], vel[0])
        else:
            base_angle = 0
        sensor_dirs = [
            np.array([np.cos(base_angle), np.sin(base_angle)]),  # front
            np.array([np.cos(base_angle + np.pi/2), np.sin(base_angle + np.pi/2)]),  # left
            np.array([np.cos(base_angle - np.pi/2), np.sin(base_angle - np.pi/2)]),  # right
            np.array([np.cos(base_angle + np.pi), np.sin(base_angle + np.pi)])  # back
        ]
        sensor_dists = [get_sensor_distance(pos, d) for d in sensor_dirs]
        # Use all four sensors for advanced fuzzy avoidance
        fuzzy_out = fuzzy.get_advanced_avoidance_policy(sensor_dists)
        # Boids logic (formation)
        neighbors = get_neighbors(i, drones, velocities)
        # Cohesion
        if neighbors:
            center = np.mean([p for _, p, _, _ in neighbors], axis=0)
            cohesion_vec = center - pos
        else:
            cohesion_vec = np.zeros(2)
        # Alignment
        if neighbors:
            avg_vel = np.mean([v for _, _, v, _ in neighbors], axis=0)
            alignment_vec = avg_vel - vel
        else:
            alignment_vec = np.zeros(2)
        # Separation
        separation_vec = np.zeros(2)
        for _, p, _, dist in neighbors:
            if dist < SAFE_DISTANCE:
                diff = pos - p
                if np.linalg.norm(diff) > 1e-6:
                    separation_vec += diff / (dist * dist)
        # Goal seeking
        goal_vec = GOAL - pos
        # Combine all
        steer = (WEIGHT_COHESION * cohesion_vec +
                 WEIGHT_ALIGNMENT * alignment_vec +
                 WEIGHT_SEPARATION * separation_vec +
                 WEIGHT_GOAL * goal_vec)
        # Normalize and limit
        if np.linalg.norm(steer) > 0:
            steer = steer / (np.linalg.norm(steer) + 1e-6)
        # Combine with fuzzy (obstacle avoidance)
        angle = base_angle + fuzzy_out['yaw_rate'] * 0.1
        move_vec = np.array([np.cos(angle), np.sin(angle)])
        # Weighted sum: more weight to obstacle avoidance if close
        w_fuzzy = max(0.5, 1.0 - sensor_dists[0] / SENSOR_RANGE)
        w_boids = 1.0 - w_fuzzy
        combined = w_fuzzy * move_vec + w_boids * steer
        if np.linalg.norm(combined) > 0:
            combined = combined / (np.linalg.norm(combined) + 1e-6)
        speed = 0.2 + fuzzy_out['thrust_adjustment'] * 0.1
        velocities[i] = combined * speed
        drones[i] += velocities[i]
        # Draw drone
        plt.plot(pos[0], pos[1], 'bo')
        # Draw sensor rays
        colors = ['r--', 'g--', 'b--', 'y--']
        for d, dist, c in zip(sensor_dirs, sensor_dists, colors):
            ray_end = pos + d * dist
            plt.plot([pos[0], ray_end[0]], [pos[1], ray_end[1]], c)
    plt.xlim(0, WORLD_SIZE)
    plt.ylim(0, WORLD_SIZE)
    plt.title(f'Step {frame}')

if __name__ == '__main__':
    fig = plt.figure(figsize=(6,6))
    ani = FuncAnimation(fig, update, frames=STEPS, interval=100)
    plt.show()
