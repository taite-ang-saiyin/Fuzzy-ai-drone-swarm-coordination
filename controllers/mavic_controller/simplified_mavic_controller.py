import sys
import os
import numpy as np
import random

# Go up from controllers/mavic_controller/ to project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

from controller import Robot
from agents.comm import UdpPeer
from agents import boids
from agents.config import get_config
from agents.fuzzy_controller import FuzzyController
from agents.goals import WaypointManager

# Get configuration
C = get_config()

# === Initialization ===
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# --- Unique drone ID from Webots node name ---
DRONE_ID = robot.getName().lower()
try:
    idx = int(DRONE_ID.replace("drone", ""))
except Exception:
    idx = 0

# --- UDP ports ---
base_port = C['comms']['base_port']
num_drones = 5
LISTEN_PORT = base_port + idx
PEER_PORTS = [base_port + i for i in range(num_drones) if i != idx]

# === Get devices ===
def try_get_and_enable(device_name):
    d = robot.getDevice(device_name)
    if d is not None:
        try:
            d.enable(timestep)
        except Exception:
            pass
    return d

imu = try_get_and_enable("inertial unit")
gps = try_get_and_enable("gps")
gyro = try_get_and_enable("gyro")
compass = try_get_and_enable("compass")
camera = try_get_and_enable("camera")

# Distance sensors
front_ds = try_get_and_enable("front distance sensor") or try_get_and_enable("front range finder")
back_ds = try_get_and_enable("back distance sensor")
left_ds = try_get_and_enable("left distance sensor")
right_ds = try_get_and_enable("right distance sensor")

# Motors
front_left_motor = robot.getDevice("front left propeller")
front_right_motor = robot.getDevice("front right propeller")
rear_left_motor = robot.getDevice("rear left propeller")
rear_right_motor = robot.getDevice("rear right propeller")
motors = [front_left_motor, front_right_motor, rear_left_motor, rear_right_motor]

# Set motors to velocity mode - start with zero velocity for stable takeoff
for motor in motors:
    motor.setPosition(float("inf"))
    motor.setVelocity(0.0)  # Start with zero velocity to prevent immediate thrust

# === Control constants ===
K_VERTICAL_THRUST = 70.0
K_VERTICAL_OFFSET = 0.6
K_VERTICAL_P = 3.0
K_ROLL_P = 50.0
K_PITCH_P = 30.0
target_altitude = 1.0

# === Communication ===
comm = UdpPeer(DRONE_ID, LISTEN_PORT, PEER_PORTS)
comm.start()

# === Controllers ===
fuzzy_controller = FuzzyController()
waypoint_manager = WaypointManager(global_waypoints=[
    np.array([20.0, 20.0, 1.0]),
    np.array([-20.0, 20.0, 1.0]),
    np.array([-20.0, -20.0, 1.0]),
    np.array([20.0, -20.0, 1.0])
])

# === Velocity computation ===
last_pos = np.array(gps.getValues()) if gps is not None else np.zeros(3)
last_time = robot.getTime()

def get_observation():
    """Get observation for decision making"""
    # Position and orientation
    position = np.array(gps.getValues()) if gps is not None else np.zeros(3)
    roll, pitch, yaw = imu.getRollPitchYaw() if imu is not None else (0.0, 0.0, 0.0)
    
    # Distance sensor readings (normalized to 0-1)
    def safe_ds(ds):
        try:
            return float(ds.getValue()) / 1000.0  # Convert mm to meters, then normalize
        except Exception:
            return 1.0

    front_dist = safe_ds(front_ds) if front_ds is not None else 1.0
    back_dist = safe_ds(back_ds) if back_ds is not None else 1.0
    left_dist = safe_ds(left_ds) if left_ds is not None else 1.0
    right_dist = safe_ds(right_ds) if right_ds is not None else 1.0
    
    # Normalize distances (assuming max range of 10m)
    max_range = 10.0
    distances = [
        min(front_dist / max_range, 1.0),
        min(back_dist / max_range, 1.0),
        min(left_dist / max_range, 1.0),
        min(right_dist / max_range, 1.0)
    ]
    
    # Neighbor information
    world = comm.snapshot()
    neighbor_positions = [np.array(s["pos"]) for peer_id, s in world.items() if peer_id != DRONE_ID]
    
    # Print comprehensive distance matrix between all drones every 2 seconds
    if robot.getTime() % 2.0 < 0.1:  # Print every 2 seconds
        print(f"\n=== DRONE DISTANCE MATRIX (Time: {robot.getTime():.1f}s) ===")
        
        # Get all drone positions including current drone
        all_drone_positions = {DRONE_ID: position}
        for peer_id, s in world.items():
            all_drone_positions[peer_id] = np.array(s["pos"])
        
        # Create distance matrix
        drone_ids = sorted(all_drone_positions.keys())
        print("     ", end="")
        for drone_id in drone_ids:
            print(f"{drone_id:>8}", end="")
        print()
        
        for i, drone_id1 in enumerate(drone_ids):
            print(f"{drone_id1:>5}", end="")
            for j, drone_id2 in enumerate(drone_ids):
                if i == j:
                    print(f"{'0.00':>8}", end="")
                else:
                    pos1 = all_drone_positions[drone_id1]
                    pos2 = all_drone_positions[drone_id2]
                    dist = np.linalg.norm(pos1 - pos2)
                    print(f"{dist:>8.2f}", end="")
            print()
        
        # Print individual drone details
        print(f"\n--- Individual Drone Details ---")
        for drone_id in drone_ids:
            pos = all_drone_positions[drone_id]
            print(f"{drone_id}: Position ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
        
        print("=" * 50)
    
    neighbor_distances = [np.linalg.norm(position - p) for p in neighbor_positions] if neighbor_positions else [10.0]
    nearest_neighbor_distance = min(neighbor_distances) if neighbor_distances else 10.0
    nearest_neighbor_distance = min(nearest_neighbor_distance / max_range, 1.0)  # Normalize
    
    # Get goal (shared for all drones)
    goal = waypoint_manager.get_global_goal()
    
    # Determine if this is the leader (drone0) or follower
    is_leader = (DRONE_ID == 'drone0')
    
    return {
        'position': position,
        'orientation': [roll, pitch, yaw],
        'distances': distances,
        'nearest_neighbor_distance': nearest_neighbor_distance,
        'goal': goal,
        'is_leader': is_leader
    }

# === Main simulation loop ===
while robot.step(timestep) != -1:
    # Get observation
    observation = get_observation()
    position = observation['position']
    distances = observation['distances']
    nearest_neighbor_distance = observation['nearest_neighbor_distance']
    goal = observation['goal']
    
    # Apply fuzzy controller for obstacle avoidance
    fuzzy_output = fuzzy_controller.get_advanced_avoidance_policy(
        distances, 
        nearest_neighbor_distance, 
        drone_id=DRONE_ID
    )
    
    # Get boids acceleration for formation
    current_time = robot.getTime()
    dt = current_time - last_time
    if dt <= 1e-6:
        dt = 1e-3
    
    current_pos = np.array(gps.getValues()) if gps is not None else last_pos
    vel = (current_pos - last_pos) / dt
    last_pos, last_time = current_pos, current_time
    
    # Broadcast state and get boids acceleration
    comm.broadcast(current_pos, vel)
    world = comm.snapshot()
    boids_acc = boids.step(current_pos, vel, world, goal=goal)
    
    # Combine fuzzy avoidance with boids formation
    thrust_adj = fuzzy_output.get('thrust_adjustment', 0.0)
    yaw_rate = fuzzy_output.get('yaw_rate', 0.0)
    roll_adj = fuzzy_output.get('roll_adjustment', 0.0)
    
    # === STOP-AND-WAIT MECHANISM ===
    # If any drone is within 2m, implement stop-and-wait behavior
    MIN_SAFE_DISTANCE = 2.0  # meters
    should_stop = False
    closest_drone_id = None
    
    # Get neighbor distances
    neighbor_distances = []
    for peer_id, s in world.items():
        if peer_id != DRONE_ID:
            neighbor_pos = np.array(s["pos"])
            dist = np.linalg.norm(position - neighbor_pos)
            neighbor_distances.append((peer_id, dist))
    
    if neighbor_distances:
        # Find closest neighbor
        closest_drone_id, min_distance = min(neighbor_distances, key=lambda x: x[1])
        
        if min_distance < MIN_SAFE_DISTANCE:
            # Get current goal to determine which drone is "front" (closer to goal)
            current_goal = waypoint_manager.get_global_goal()
            current_distance_to_goal = np.linalg.norm(position - current_goal)
            
            # Get closest drone's position and distance to goal
            closest_drone_pos = np.array(world[closest_drone_id]['pos'])
            closest_drone_distance_to_goal = np.linalg.norm(closest_drone_pos - current_goal)
            
            # Front drone (closer to goal) continues, rear drone (further from goal) stops
            if current_distance_to_goal < closest_drone_distance_to_goal:
                # This drone is closer to goal (front) - continue
                should_stop = False
                print(f"[{DRONE_ID}] CONTINUE: Too close to {closest_drone_id} ({min_distance:.2f}m < {MIN_SAFE_DISTANCE}m). This drone is front (closer to goal), continuing.")
            else:
                # This drone is further from goal (rear) - stop
                should_stop = True
                print(f"[{DRONE_ID}] STOP-AND-WAIT: Too close to {closest_drone_id} ({min_distance:.2f}m < {MIN_SAFE_DISTANCE}m). This drone is rear (further from goal), stopping.")
    
    # Calculate goal-seeking
    goal_vector = goal - position
    distance_to_goal = np.linalg.norm(goal_vector[:2])
    
    # Leader-follower behavior
    is_leader = observation['is_leader']
    
    if is_leader:
        # Leader: move directly toward goal
        if distance_to_goal > 0.1:
            goal_direction = goal_vector / np.linalg.norm(goal_vector)
            goal_thrust = min(1.0, distance_to_goal / 10.0)
        else:
            goal_direction = np.zeros(3)
            goal_thrust = 0.0
        
        # Leader prioritizes goal-seeking
        goal_weight = 0.7
        formation_weight = 0.2
        avoidance_weight = 0.1
        
        pitch_disturbance = (goal_direction[0] * goal_thrust * 0.3 + thrust_adj * 0.1)
        roll_disturbance = (boids_acc[1] * 0.05 + roll_adj * 0.1)
        yaw_disturbance = (boids_acc[2] * 0.1 + yaw_rate * 0.4)
        
        # Apply 2x speed boost for leader drone
        pitch_disturbance *= 2.0
        
    else:
        # Follower: maintain formation relative to leader
        leader_pos = None
        leader_vel = None
        
        # Find leader in world states
        for peer_id, state in world.items():
            if peer_id == 'drone0':  # Leader
                leader_pos = np.array(state['pos'])
                leader_vel = np.array(state['vel'])
                break
        
        if leader_pos is not None:
            # Calculate formation position relative to leader
            formation_spacing = 4.0
            formation_angle = np.radians(30)
            
            # Get leader's heading - use goal direction if velocity is too low
            if np.linalg.norm(leader_vel) > 0.1:
                leader_heading = np.arctan2(leader_vel[1], leader_vel[0])
            else:
                # If leader velocity is too low, use goal direction for heading
                goal_vector = goal - leader_pos
                if np.linalg.norm(goal_vector[:2]) > 1e-3:
                    leader_heading = np.arctan2(goal_vector[1], goal_vector[0])
                else:
                    leader_heading = 0.0
            
            # Calculate formation offset based on drone index
            drone_idx = int(DRONE_ID.replace('drone', ''))
            side = -1 if drone_idx % 2 == 1 else 1
            rank = (drone_idx + 1) // 2
            
            # Formation position relative to leader
            dx = -rank * formation_spacing * np.cos(formation_angle)
            dy = side * rank * formation_spacing * np.sin(formation_angle)
            
            # Rotate offset by leader's heading
            cos_h = np.cos(leader_heading)
            sin_h = np.sin(leader_heading)
            offset_x = dx * cos_h - dy * sin_h
            offset_y = dx * sin_h + dy * cos_h
            
            desired_pos = leader_pos + np.array([offset_x, offset_y, 0])
            formation_vector = desired_pos - position
            
            # Follower prioritizes formation
            formation_weight = 0.6
            goal_weight = 0.2
            avoidance_weight = 0.2
            
            if np.linalg.norm(formation_vector) > 0:
                formation_direction = formation_vector / np.linalg.norm(formation_vector)
            else:
                formation_direction = np.zeros(3)
            
            pitch_disturbance = (formation_direction[0] * 0.2 + thrust_adj * 0.1)
            roll_disturbance = (formation_direction[1] * 0.2 + roll_adj * 0.1)
            yaw_disturbance = (formation_direction[2] * 0.1 + yaw_rate * 0.3)
            
            # Apply 1.6x speed boost for follower drones
            pitch_disturbance *= 1.6
        else:
            # No leader found, fall back to goal-seeking
            if distance_to_goal > 0.1:
                goal_direction = goal_vector / np.linalg.norm(goal_vector)
                goal_thrust = min(1.0, distance_to_goal / 10.0)
            else:
                goal_direction = np.zeros(3)
                goal_thrust = 0.0
            
            pitch_disturbance = (goal_direction[0] * goal_thrust * 0.2 + thrust_adj * 0.1)
            roll_disturbance = (roll_adj * 0.1)
            yaw_disturbance = (yaw_rate * 0.4)
            
            # Apply 1.6x speed boost for follower drones (fallback mode)
            pitch_disturbance *= 1.6
    
    # Read sensors for PID control
    roll, pitch, yaw = imu.getRollPitchYaw() if imu is not None else (0.0, 0.0, 0.0)
    altitude = position[2] if gps is not None else 0.0
    roll_velocity = gyro.getValues()[0] if gyro is not None else 0.0
    pitch_velocity = gyro.getValues()[1] if gyro is not None else 0.0
    
    # Apply stop-and-wait mechanism if too close to other drones
    if should_stop:
        pitch_disturbance = 0.0  # Stop all forward/backward movement
        print(f"[{DRONE_ID}] [STOP] Drone stopped due to proximity to {closest_drone_id}")
    
    # PID-like stabilization
    roll_input = K_ROLL_P * max(-1.0, min(1.0, roll)) + roll_velocity + roll_disturbance
    pitch_input = K_PITCH_P * max(-1.0, min(1.0, pitch)) + pitch_velocity + pitch_disturbance
    clamped_diff_alt = max(-1.0, min(1.0, target_altitude - altitude + K_VERTICAL_OFFSET))
    vertical_input = K_VERTICAL_P * (clamped_diff_alt ** 3.0)
    
    # Calculate motor speeds
    fl = K_VERTICAL_THRUST + vertical_input - roll_input + pitch_input - yaw_disturbance
    fr = K_VERTICAL_THRUST + vertical_input + roll_input + pitch_input + yaw_disturbance
    rl = K_VERTICAL_THRUST + vertical_input - roll_input - pitch_input + yaw_disturbance
    rr = K_VERTICAL_THRUST + vertical_input + roll_input - pitch_input - yaw_disturbance
    
    # Safety clamp: prevent NaN, infinite values, and excessive speeds
    motor_speeds = [fl, fr, rl, rr]
    motor_speeds = [0.0 if not np.isfinite(v) else v for v in motor_speeds]
    
    # Additional safety limits to prevent crashes
    MAX_MOTOR_SPEED = 100.0  # Maximum safe motor speed
    MIN_MOTOR_SPEED = 0.0    # Minimum motor speed (can't go negative)
    motor_speeds = [max(MIN_MOTOR_SPEED, min(MAX_MOTOR_SPEED, v)) for v in motor_speeds]
    
    # Set motor speeds with correct signs based on PROTO thrust constants
    front_left_motor.setVelocity(motor_speeds[0])      # FL: positive velocity for positive thrust
    front_right_motor.setVelocity(-motor_speeds[1])    # FR: negative velocity for positive thrust
    rear_left_motor.setVelocity(-motor_speeds[2])      # RL: negative velocity for positive thrust
    rear_right_motor.setVelocity(motor_speeds[3])      # RR: positive velocity for positive thrust
    
    # Check if goal reached
    if distance_to_goal < 2.0:
        waypoint_manager.global_goal_reached(position, thresh=2.0)
    
    # Debug output
    if robot.getTime() % 2.0 < 0.1:  # Print every 2 seconds
        print(f"[{DRONE_ID}] Pos: {position[:2]}, Goal: {goal[:2]}, Dist: {distance_to_goal:.1f}m")
        print(f"[{DRONE_ID}] Sensors: F={distances[0]:.2f}, B={distances[1]:.2f}, L={distances[2]:.2f}, R={distances[3]:.2f}")
        print(f"[{DRONE_ID}] Controls: T={thrust_adj:.2f}, Y={yaw_rate:.2f}, R={roll_adj:.2f}")

# Cleanup
comm.stop()
