import sys
import os
import numpy as np
import random
import struct
import json

# Go up from controllers/mavic_controller/ to project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

from controller import Robot, Receiver
from agents.comm import UdpPeer
from agents import boids
from agents.config import get_config
C = get_config()



# Import fuzzy policies
from agents.fuzzy_controller import FuzzyController

# === RL Model (optional) ===
RL_MODEL = None
RL_AVAILABLE = False
try:
    from stable_baselines3 import PPO
    RL_MODEL_PATH = os.path.join(PROJECT_ROOT, "controllers", "my_rl_supervisor", "ppo_final_model.zip")
    if os.path.exists(RL_MODEL_PATH):
        RL_MODEL = PPO.load(RL_MODEL_PATH, device="cpu")
        RL_AVAILABLE = True
        print("[RL] PPO model loaded for drone controller.")
    else:
        print("[RL] PPO model not found, running without RL.")
except Exception as e:
    print(f"[RL] RL model or stable_baselines3 not available: {e}")

# Import dynamic goal manager
from agents.goals import WaypointManager
# --- WaypointManager supports global, per-drone, and dynamic waypoints ---
goal_manager = WaypointManager()

# === Initialization ===
robot = Robot()
timestep = int(robot.getBasicTimeStep())  # ms, set to 0.2s for faster simulation


# --- Unique drone ID from Webots node name ---
DRONE_ID = robot.getName().lower()   # "drone0" … "droneN"
# print(f"[DEBUG] robot.getName() -> {DRONE_ID}")

try:
    idx = int(DRONE_ID.replace("drone", ""))
except Exception as e:
    # print(f"[ERROR] Could not parse DRONE_ID {DRONE_ID}: {e}")
    idx = 0


# Example: set a new global waypoint list dynamically
# goal_manager.set_global_waypoints([np.array([0.0, 5.0, 1.0]), np.array([0.0, -5.0, 1.0])])


# Set global waypoints for all drones (example: two waypoints)
goal_manager.set_global_waypoints([
    np.array([24.6, 0, 1.00]),  # First target position
    np.array([24.6, 14.12, 1.00]) # Second target position
])


# --- UDP ports ---
base_port = C['comms']['base_port']
num_drones = 5  # Match the actual number of drones
LISTEN_PORT = base_port + idx
PEER_PORTS = [base_port + i for i in range(num_drones) if i != idx]

if DRONE_ID == 'drone2':
    print(f"[INIT] {DRONE_ID} → idx={idx}, LISTEN_PORT={LISTEN_PORT}, PEERS={PEER_PORTS}")

def try_get_and_enable(device_name):
    d = robot.getDevice(device_name)
    if d is not None:
        try:
            d.enable(timestep)
        except Exception:
            pass
    return d

# === Get devices ===
imu = try_get_and_enable("inertial unit")
gps = try_get_and_enable("gps")
gyro = try_get_and_enable("gyro")
compass = try_get_and_enable("compass")

# Camera (vision-based obstacle estimation)
camera = try_get_and_enable("camera")

# Distance sensors (try multiple common names, fall back to None)
front_ds = try_get_and_enable("front distance sensor") or try_get_and_enable("front range finder") or try_get_and_enable("range finder")
back_ds = try_get_and_enable("back distance sensor")
left_ds = try_get_and_enable("left distance sensor")
right_ds = try_get_and_enable("right distance sensor")

# Receiver/Emitter for supervisor commands/telemetry (may not exist in PROTO)
receiver = robot.getDevice("receiver")
if receiver is not None:
    try:
        receiver.enable(timestep)
    except Exception:
        pass
emitter = robot.getDevice("emitter") if hasattr(robot, "getDevice") else None

front_left_motor  = robot.getDevice("front left propeller")
front_right_motor = robot.getDevice("front right propeller")
rear_left_motor   = robot.getDevice("rear left propeller")
rear_right_motor  = robot.getDevice("rear right propeller")
motors = [front_left_motor, front_right_motor, rear_left_motor, rear_right_motor]

# Set motors to velocity mode
for motor in motors:
    motor.setPosition(float("inf"))
    motor.setVelocity(1.0)

if DRONE_ID == 'drone2':
    print(f"[{DRONE_ID}] Mavic2Pro AI controller started!")

# === PID / Control constants ===
K_VERTICAL_THRUST = 70.0
K_VERTICAL_OFFSET = 0.6
K_VERTICAL_P = 4.0  # Increased for faster altitude control
K_ROLL_P = 60.0     # Increased for faster roll response
K_PITCH_P = 40.0    # Increased for faster pitch response

target_altitude = 1.0

# Speed scaling: reduced to 1x for safer drone movement
SPEED_MULTIPLIER = 1.0

# === Communication ===
comm = UdpPeer(DRONE_ID, LISTEN_PORT, PEER_PORTS)
comm.start()
if DRONE_ID == 'drone2':
    print(f"[{DRONE_ID}] UdpPeer started, listening on {LISTEN_PORT}")


# === AI Control Variables ===
fuzzy_controller = FuzzyController()
current_action = 0
thrust_adjustment = 0.0
yaw_rate = 0.0

# === Velocity computation (for boids) ===
last_pos = np.array(gps.getValues())
last_time = robot.getTime()
latest_command = {"goal": None, "velocity": None, "mode": None}
last_command_time = -1.0

def get_observation():
    """Get observation for AI decision making"""
    # Position and orientation
    position = np.array(gps.getValues()) if gps is not None else np.zeros(3)
    roll, pitch, yaw = imu.getRollPitchYaw() if imu is not None else (0.0, 0.0, 0.0)
    
    # Velocity
    # Webots Robot has no getVelocity(), use GPS delta as velocity
    velocity = np.zeros(3)
    
    # Distance sensor readings (normalized)
    def safe_ds(ds):
        try:
            return float(ds.getValue()) / 1000.0  # Convert mm to meters
        except Exception:
            return 1.0

    # Start with distance sensors if any (all handled the same way)
    front_dist = safe_ds(front_ds)
    front_source = 'sensor' if front_ds is not None else 'none'
    back_dist = safe_ds(back_ds)
    left_dist = safe_ds(left_ds)
    right_dist = safe_ds(right_ds)
    
    # Compass heading
    if compass is not None:
        compass_values = compass.getValues()
        heading = np.arctan2(compass_values[0], compass_values[1])
    else:
        heading = 0.0
    
    # Neighbor information from boids
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
    neighbor_count = len(neighbor_positions)


    # --- Unified Leader-Follower Formation Logic ---
    # Leader approaches target directly, followers only maintain formation with leader
    shared_goal = goal_manager.get_global_goal()
    is_leader = (DRONE_ID == 'drone0')
    
    if is_leader:
        # Leader: use the shared goal directly (approaches target)
        goal = shared_goal
        goal_dx, goal_dy, goal_dz = goal - position
    else:
        # Follower: calculate formation position relative to leader (not direct goal)
        goal = shared_goal  # Keep for reference, but won't be used for direct navigation
        goal_dx, goal_dy, goal_dz = goal - position  # This will be overwritten with formation vector
        
        # Get leader information for formation
        world_snapshot = comm.snapshot()
        leader_pos = None
        leader_vel = None
        if 'drone0' in world_snapshot:
            leader_pos = np.array(world_snapshot['drone0']['pos'])
            leader_vel = np.array(world_snapshot['drone0']['vel'])
        else:
            # Leader not found - fallback to goal-seeking
            print(f"[{DRONE_ID}] WARNING: Leader (drone0) not found in communication! Falling back to goal-seeking.")
        
        # Store leader info for later use in control logic
        if leader_pos is not None:
            # Calculate desired formation position relative to leader
            formation_spacing = 4.0  # meters between drones in the V
            formation_angle = np.radians(30)  # V angle
            
            # Get leader's heading - use goal direction if velocity is too low
            if np.linalg.norm(leader_vel[:2]) > 1e-3:
                leader_heading = np.arctan2(leader_vel[1], leader_vel[0])
            else:
                # If leader velocity is too low, use goal direction for heading
                goal_vector = shared_goal - leader_pos
                if np.linalg.norm(goal_vector[:2]) > 1e-3:
                    leader_heading = np.arctan2(goal_vector[1], goal_vector[0])
                else:
                    leader_heading = 0.0
            
            # Calculate formation offset based on drone index
            side = -1 if idx % 2 == 1 else 1  # left for odd, right for even
            rank = (idx + 1) // 2
            
            # Formation position relative to leader
            dx = -rank * formation_spacing * np.cos(formation_angle)
            dy = side * rank * formation_spacing * np.sin(formation_angle)
            
            # Rotate offset by leader's heading
            cos_h = np.cos(leader_heading)
            sin_h = np.sin(leader_heading)
            offset_x = dx * cos_h - dy * sin_h
            offset_y = dx * sin_h + dy * cos_h
            
            desired_formation_pos = leader_pos + np.array([offset_x, offset_y, 0.0])
            formation_vector = desired_formation_pos - position
            
            # Debug logging for formation following
            if DRONE_ID in ['drone1', 'drone2'] and robot.getTime() % 2.0 < 0.1:
                print(f"[{DRONE_ID}] FORMATION DEBUG:")
                print(f"  Leader pos: {leader_pos[:2]}, vel: {leader_vel[:2]}")
                print(f"  Leader heading: {np.degrees(leader_heading):.1f}°")
                print(f"  Formation offset: ({offset_x:.2f}, {offset_y:.2f})")
                print(f"  Desired pos: {desired_formation_pos[:2]}")
                print(f"  Formation vector: {formation_vector[:2]}")
                print(f"  Formation distance: {np.linalg.norm(formation_vector):.2f}m")
            
            # Store formation info for control logic
            goal_dx, goal_dy, goal_dz = formation_vector

    return {
        'position': position,
        'velocity': velocity,
        'orientation': [roll, pitch, yaw],
        'distances': [front_dist, back_dist, left_dist, right_dist],
        'front_source': front_source,
        'heading': heading,
        'neighbor_count': neighbor_count,
        'nearest_neighbor_distance': nearest_neighbor_distance,
        'goal_dx': goal_dx,
        'goal_dy': goal_dy,
        'goal_dz': goal_dz,
        'goal': goal
    }


# --- RL/Fuzzy Policy Selection ---
RL_USE = getattr(C.get('rl', {}), 'enabled', False)  # Set to True in config to use RL
def apply_policy(action, observation):
    """Apply RL or fuzzy policy based on config and availability"""
    global thrust_adjustment, yaw_rate
    if RL_USE and RL_AVAILABLE and RL_MODEL is not None:
        # RL expects a 2D array for obs
        obs_vec = np.array([
            float(np.linalg.norm(observation['velocity']) / 5.0),
            float(observation['distances'][0]),
            float(observation['neighbor_count']) / 10.0,
            float(np.clip(observation['position'][2] / 5.0, 0.0, 1.0))
        ], dtype=np.float32)
        rl_action, _ = RL_MODEL.predict(obs_vec, deterministic=True)
        # Map RL action to thrust/yaw (customize as needed)
        # Example: 0=forward, 1=left, 2=right, 3=stop
        if rl_action == 0:
            thrust_adjustment, yaw_rate = 0.5, 0.0
        elif rl_action == 1:
            thrust_adjustment, yaw_rate = 0.0, 0.5
        elif rl_action == 2:
            thrust_adjustment, yaw_rate = 0.0, -0.5
        else:
            thrust_adjustment, yaw_rate = 0.0, 0.0
        if DRONE_ID == 'drone2':
            print(f"[DEBUG][RL] RL outputs: action={rl_action}, thrust_adjustment={thrust_adjustment}, yaw_rate={yaw_rate}")
        return thrust_adjustment, yaw_rate
    else:
        # Use fuzzy/boids logic
        distances = observation['distances']  # [front, back, left, right], normalized 0-1
        neighbor_distance = observation.get('nearest_neighbor_distance', 1.0)  # normalized 0-1 (max 1)
        outputs = fuzzy_controller.get_advanced_avoidance_policy(distances, neighbor_distance, drone_id=DRONE_ID)
        thrust_adjustment = outputs.get('thrust_adjustment', 0.0)
        yaw_rate = outputs.get('yaw_rate', 0.0)
        if DRONE_ID == 'drone2':
            print(f"[DEBUG][Fuzzy] Fuzzy outputs: thrust_adjustment={thrust_adjustment}, yaw_rate={yaw_rate}")
        return thrust_adjustment, yaw_rate

def check_receiver():
    """
    Check for supervisor/AI commands on the Webots receiver.
    Supports JSON payloads:
      {"id": "drone0", "target_pos": [x,y,z], "target_vel": [vx,vy,vz], "mode": "hold"}
    Falls back to legacy struct floats (thrust, yaw) for compatibility.
    """
    global latest_command, last_command_time

    if receiver is None:
        return None

    while receiver.getQueueLength() > 0:
        data = receiver.getData()
        try:
            msg = json.loads(data.decode("utf-8"))
            target_id = msg.get("id")
            if target_id is None or target_id == DRONE_ID:
                latest_command = {
                    "goal": np.array(msg["target_pos"]) if "target_pos" in msg else latest_command["goal"],
                    "velocity": np.array(msg["target_vel"]) if "target_vel" in msg else latest_command["velocity"],
                    "mode": msg.get("mode", latest_command["mode"])
                }
                last_command_time = robot.getTime()
                if DRONE_ID == 'drone2':
                    print(f"[{DRONE_ID}] Received JSON cmd: {msg}")
        except Exception:
            # Legacy binary: thrust, yaw
            try:
                if len(data) >= 8:
                    thrust, yaw = struct.unpack('ff', data[:8])
                    latest_command["velocity"] = None  # legacy doesn't set velocity
                    latest_command["goal"] = None
                    latest_command["mode"] = None
                    last_command_time = robot.getTime()
                    if DRONE_ID == 'drone2':
                        print(f"[{DRONE_ID}] Received legacy thrust/yaw: {thrust}, {yaw}")
                else:
                    if DRONE_ID == 'drone2':
                        print(f"[{DRONE_ID}] Received data too short: {len(data)} bytes")
            except Exception as e:
                if DRONE_ID == 'drone2':
                    print(f"[{DRONE_ID}] Receiver unpack failed: {e}")
        receiver.nextPacket()
    return None


def send_telemetry(position, velocity, distances, goal):
    """Lightweight telemetry back to supervisor via emitter (if present)."""
    if emitter is None:
        return
    if robot.getTime() % 0.5 > (timestep / 1000.0):  # ~2 Hz
        return
    try:
        payload = {
            "id": DRONE_ID,
            "t": robot.getTime(),
            "pos": position.tolist(),
            "vel": velocity.tolist(),
            "distances": distances,
            "goal": goal.tolist() if goal is not None else None
        }
        emitter.send(json.dumps(payload).encode("utf-8"))
    except Exception:
        pass


# === Main simulation loop ===
while robot.step(timestep) != -1:
    # --- Check if all drones have taken off ---
    # Gather all drone altitudes from comm.snapshot()
    min_flight_altitude = 0.5
    world = comm.snapshot()
    # Use current altitude from above (already defined)
    all_altitudes = [s['pos'][2] for peer_id, s in world.items()]
    all_altitudes.append(gps.getValues()[2] if gps is not None else 0.0)  # include self
    all_taken_off = all(a > min_flight_altitude for a in all_altitudes)
    # --- Check for AI commands ---

    # --- Unified Waypoint Management ---
    # Only the leader advances waypoints, all drones use the same shared goal
    obs_pos = None
    if gps is not None:
        obs_pos = np.array(gps.getValues())
    
    # Determine if this drone is the leader
    is_leader = (DRONE_ID == 'drone0')
    
    if obs_pos is not None and is_leader:
        # Only leader checks if goal is reached and advances waypoints
        current_goal = goal_manager.get_global_goal()
        distance_to_goal = np.linalg.norm(obs_pos - current_goal)
        
        if distance_to_goal < 3.0:  # Goal reached threshold
            goal_manager.global_goal_reached(obs_pos, thresh=3.0)
            new_goal = goal_manager.get_global_goal()
            print(f"[LEADER] Goal reached! Moving to next waypoint: {new_goal}")
        else:
            print(f"[LEADER] Position: {obs_pos[:2]}, Goal: {current_goal[:2]}, Distance: {distance_to_goal:.1f}m")
    check_receiver()

    # --- Read sensors ---
    roll, pitch, yaw = imu.getRollPitchYaw() if imu is not None else (0.0, 0.0, 0.0)
    altitude = (gps.getValues()[2] if gps is not None else 0.0)
    roll_velocity = (gyro.getValues()[0] if gyro is not None else 0.0)
    pitch_velocity = (gyro.getValues()[1] if gyro is not None else 0.0)

    # --- Get observation for AI ---
    observation = get_observation()

    # Apply fresh supervisor command if available
    cmd_goal = None
    cmd_vel = None
    if last_command_time >= 0 and (robot.getTime() - last_command_time) < 2.0:
        cmd_goal = latest_command.get("goal")
        cmd_vel = latest_command.get("velocity")
    if cmd_goal is not None:
        observation['goal'] = cmd_goal
    # Print sensor data every step for debugging
    print(f"[{DRONE_ID}] SENSOR DEBUG: front={observation['distances'][0]:.2f}, back={observation['distances'][1]:.2f}, left={observation['distances'][2]:.2f}, right={observation['distances'][3]:.2f}")



    # --- Leader-Follower Control Logic ---
    # Determine if this is the leader or follower
    is_leader = (DRONE_ID == 'drone0')
    
    # Get fuzzy avoidance outputs
    distances = observation['distances']
    neighbor_distance = observation.get('nearest_neighbor_distance', 1.0)
    fuzzy_outputs = fuzzy_controller.get_advanced_avoidance_policy(distances, neighbor_distance, drone_id=DRONE_ID)
    fuzzy_thrust = fuzzy_outputs.get('thrust_adjustment', 0.0)
    fuzzy_yaw = fuzzy_outputs.get('yaw_rate', 0.0)
    
    # Get goal information (override with supervisor command if present)
    goal = observation['goal']
    goal_dx, goal_dy, goal_dz = goal - observation['position']
    distance_to_goal = np.linalg.norm([goal_dx, goal_dy, goal_dz])
    
    # === STOP-AND-WAIT MECHANISM ===
    # If any drone is within 2m, implement stop-and-wait behavior
    MIN_SAFE_DISTANCE = 2.0  # meters
    should_stop = False
    closest_drone_id = None
    
    # Get neighbor distances from observation
    neighbor_distances = []
    world_snapshot = comm.snapshot()
    current_position = observation['position']
    
    for peer_id, s in world_snapshot.items():
        if peer_id != DRONE_ID:
            neighbor_pos = np.array(s["pos"])
            dist = np.linalg.norm(current_position - neighbor_pos)
            neighbor_distances.append((peer_id, dist))
    
    if neighbor_distances:
        # Find closest neighbor
        closest_drone_id, min_distance = min(neighbor_distances, key=lambda x: x[1])
        
        if min_distance < MIN_SAFE_DISTANCE:
            # Get current goal to determine which drone is "front" (closer to goal)
            current_goal = goal_manager.get_global_goal()
            current_distance_to_goal = np.linalg.norm(current_position - current_goal)
            
            # Get closest drone's position and distance to goal
            closest_drone_pos = np.array(world_snapshot[closest_drone_id]['pos'])
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

    # Supervisor hold mode overrides motion briefly after command receipt
    if (robot.getTime() - last_command_time) < 2.0 and latest_command.get("mode") == "hold":
        should_stop = True
    
    if is_leader:
        # Leader: prioritize goal-seeking with obstacle avoidance
        goal_weight = 0.7
        avoidance_weight = 0.3
        
        # Calculate goal-seeking thrust and yaw
        if distance_to_goal > 0.05:  # Reduced threshold to prevent stopping
            goal_direction = np.array([goal_dx, goal_dy, goal_dz]) / distance_to_goal
            
            # Adaptive goal-seeking thrust - slow down when avoiding obstacles
            # Note: sensor values are normalized 0-1, where 1.0 = no obstacle, 0.0 = collision
            min_obstacle_distance = min(observation['distances'])
            if min_obstacle_distance < 0.3:  # Very close obstacle (0.0-0.3)
                goal_thrust = min(0.3, distance_to_goal / 25.0)  # Much slower for safety
                print(f"[{DRONE_ID}] [GOAL] Obstacle very close (dist={min_obstacle_distance:.2f}), using very slow goal-seeking")
            elif min_obstacle_distance < 0.6:  # Close obstacle (0.3-0.6)
                goal_thrust = min(0.6, distance_to_goal / 20.0)  # Moderate speed
                print(f"[{DRONE_ID}] [GOAL] Obstacle nearby (dist={min_obstacle_distance:.2f}), using slow goal-seeking")
            else:  # Clear path (0.6-1.0)
                goal_thrust = min(0.8, distance_to_goal / 15.0)  # Conservative for clear path
                print(f"[{DRONE_ID}] [GOAL] Clear path (dist={min_obstacle_distance:.2f}), using conservative goal-seeking")
            
            # Calculate desired heading
            desired_heading = np.arctan2(goal_dy, goal_dx)
            current_heading = observation['heading']
            heading_error = (desired_heading - current_heading + np.pi) % (2 * np.pi) - np.pi
            goal_yaw = heading_error * 1.0  # Increased from 0.5 for more responsive turning
        else:
            # Very close to goal - use minimal movement to maintain position
            goal_thrust = 0.1  # Small thrust to maintain position
            goal_yaw = 0.0
            print(f"[{DRONE_ID}] GOAL CLOSE! Distance={distance_to_goal:.3f}m, using minimal thrust")
        
        # Blend goal-seeking with obstacle avoidance
        thrust_adjustment = goal_weight * goal_thrust + avoidance_weight * fuzzy_thrust
        
        # Apply 1.2x speed boost for leader drone
        thrust_adjustment *= 1.2
        
        # Apply stop-and-wait mechanism if too close to other drones
        if should_stop:
            thrust_adjustment = 0.0  # Stop all forward/backward movement
            print(f"[{DRONE_ID}] [STOP] Leader stopped due to proximity to {closest_drone_id}")
        
        yaw_rate = goal_weight * goal_yaw + avoidance_weight * fuzzy_yaw
        
        if DRONE_ID == 'drone0':
            print(f"[LEADER] Goal: {goal[:2]}, Dist: {distance_to_goal:.1f}m, Thrust: {thrust_adjustment:.2f}, Yaw: {yaw_rate:.2f}")
            print(f"[LEADER] Goal direction: {goal_direction[:2]}, Goal thrust: {goal_thrust:.2f}, Goal yaw: {goal_yaw:.2f}")
    
    else:
        # Follower: ONLY maintain formation with leader (no direct goal-seeking)
        formation_weight = 0.8  # Primary: maintain formation
        avoidance_weight = 0.2  # Secondary: avoid obstacles
        
        # Get formation vector from observation (calculated in get_observation)
        formation_dx = goal_dx  # This is actually the formation vector for followers
        formation_dy = goal_dy
        formation_dz = goal_dz
        
        formation_distance = np.linalg.norm([formation_dx, formation_dy, formation_dz])
        
        if formation_distance > 0.05:  # Reduced threshold to prevent stopping
            # Calculate formation-seeking thrust and yaw
            formation_direction = np.array([formation_dx, formation_dy, formation_dz]) / formation_distance
            formation_thrust = min(0.8, formation_distance / 5.0)  # Conservative formation following
            
            # Calculate desired heading for formation
            desired_heading = np.arctan2(formation_dy, formation_dx)
            current_heading = observation['heading']
            heading_error = (desired_heading - current_heading + np.pi) % (2 * np.pi) - np.pi
            formation_yaw = heading_error * 0.3  # Scale down for smoother control
        else:
            # Very close to formation position - use minimal movement to maintain position
            formation_thrust = 0.1  # Small thrust to maintain position
            formation_yaw = 0.0
            print(f"[{DRONE_ID}] FORMATION CLOSE! Distance={formation_distance:.3f}m, using minimal thrust")
        
        # Followers ONLY do formation maintenance + obstacle avoidance
        # No direct goal-seeking - they reach the target by following the leader
        thrust_adjustment = (formation_weight * formation_thrust + 
                           avoidance_weight * fuzzy_thrust)
        yaw_rate = (formation_weight * formation_yaw + 
                   avoidance_weight * fuzzy_yaw)
        
        # Apply stop-and-wait mechanism if too close to other drones
        if should_stop:
            thrust_adjustment = 0.0  # Stop all forward/backward movement
            print(f"[{DRONE_ID}] [STOP] Follower stopped due to proximity to {closest_drone_id}")
        
        # Debug output for followers
        if DRONE_ID in ['drone1', 'drone2']:  # Debug for first two followers
            print(f"[{DRONE_ID}] FORMATION ONLY! Formation: {formation_distance:.1f}m, Thrust: {thrust_adjustment:.2f}, Yaw: {yaw_rate:.2f}")

    # --- Emergency Obstacle Avoidance Override ---
    # If any obstacle is very close, prioritize avoidance over formation
    EMERGENCY_THRESHOLD = 0.8  # Increased threshold for earlier emergency response
    emergency_triggered = any(dist < EMERGENCY_THRESHOLD for dist in observation['distances'])
    
    # Check for specific obstacle emergencies
    front_emergency = observation['distances'][0] < EMERGENCY_THRESHOLD
    back_emergency = observation['distances'][1] < EMERGENCY_THRESHOLD
    left_emergency = observation['distances'][2] < EMERGENCY_THRESHOLD
    right_emergency = observation['distances'][3] < EMERGENCY_THRESHOLD
    both_sides_emergency = left_emergency and right_emergency
    
    # Check if drone is too close to ground during avoidance
    current_altitude = gps.getValues()[2] if gps is not None else 0.0
    MIN_SAFE_ALTITUDE = 1.0  # Minimum safe altitude during avoidance
    altitude_emergency = current_altitude < MIN_SAFE_ALTITUDE
    
    if emergency_triggered:
        # Override with pure obstacle avoidance
        thrust_adjustment = fuzzy_thrust
        yaw_rate = fuzzy_yaw
        emergency_roll = 0.0  # Initialize emergency roll adjustment
        
        # Special handling for specific obstacle emergencies
        if altitude_emergency:
            # Drone too close to ground - prioritize altitude gain
            thrust_adjustment = -2.5  # Strong upward thrust
            yaw_rate = 0.0  # No turning to maintain stability
            emergency_roll = 0.0  # No lateral movement
            print(f"[{DRONE_ID}] [EMERGENCY] Too close to ground! Gaining altitude. Altitude: {current_altitude:.2f}m")
        elif both_sides_emergency:
            # Both sides blocked - prioritize vertical movement and gentle forward
            thrust_adjustment = -1.5  # Strong upward thrust for vertical escape
            yaw_rate = 0.0  # No turning to avoid instability
            emergency_roll = 0.0  # No lateral movement
            print(f"[{DRONE_ID}] [EMERGENCY] Both sides blocked! Using vertical escape.")
        elif back_emergency:
            # For back obstacles, prioritize forward movement and turning away
            thrust_adjustment = -2.0  # Strong forward thrust
            yaw_rate = -1.5  # Turn away from back obstacle
            emergency_roll = 0.0  # No lateral movement for back obstacles
            print(f"[{DRONE_ID}] [EMERGENCY] Back obstacle detected! Using reverse avoidance.")
        elif left_emergency:
            # For left obstacles, move right (lateral movement only, no thrust change)
            thrust_adjustment = 0.0  # No thrust change - use lateral movement only
            yaw_rate = 0.0  # No turning - just lateral movement
            emergency_roll = -1.5  # Strong roll left to move right (INVERTED)
            print(f"[{DRONE_ID}] [EMERGENCY] Left obstacle detected! Moving right laterally (no thrust change).")
        elif right_emergency:
            # For right obstacles, move left (lateral movement only, no thrust change)
            thrust_adjustment = 0.0  # No thrust change - use lateral movement only
            yaw_rate = 0.0  # No turning - just lateral movement
            emergency_roll = 1.5  # Strong roll right to move left (INVERTED)
            print(f"[{DRONE_ID}] [EMERGENCY] Right obstacle detected! Moving left laterally (no thrust change).")
        elif front_emergency:
            # For front obstacles, accelerate backward and turn away
            thrust_adjustment = 1.5  # Backward thrust to move away from front obstacle
            yaw_rate = 1.5  # Turn away from front obstacle
            emergency_roll = 0.0  # No lateral movement for front obstacles
            print(f"[{DRONE_ID}] [EMERGENCY] Front obstacle detected! Using backward acceleration and turn avoidance.")
        else:
            print(f"[{DRONE_ID}] [EMERGENCY] Obstacle detected! Using pure avoidance.")
    # Otherwise, use the leader-follower control logic calculated above

    # --- Compute velocity from GPS ---
    current_time = robot.getTime()
    dt = current_time - last_time
    if dt <= 1e-6:   # avoid division by zero
        dt = 1e-3

    current_pos = np.array(gps.getValues()) if gps is not None else last_pos
    vel = (current_pos - last_pos) / dt
    last_pos, last_time = current_pos, current_time

    # --- Communication & Formation Coordination ---
    comm.broadcast(current_pos, vel)
    world = comm.snapshot()
    send_telemetry(current_pos, vel, observation['distances'], goal)
    
    # Use boids for additional formation coordination (secondary to leader-follower)
    if not emergency_triggered:  # Only use boids when not in emergency avoidance
        acc = boids.step(current_pos, vel, world, goal=observation['goal'])
        # Strengthen separation: if nearest neighbor is very close, add extra repulsion
        nearest_neighbor_distance = observation.get('nearest_neighbor_distance', 10.0)
        if nearest_neighbor_distance < 0.5:
            # Add strong repulsion in X direction (away from neighbor)
            acc += np.array([2.0, 0.0, 0.0])
        # Replace NaNs with 0 to protect motors
        if not np.isfinite(acc).all():
            acc = np.zeros(3)
    else:
        # During emergency, add strong separation forces to prevent collisions
        acc = np.zeros(3)
        nearest_neighbor_distance = observation.get('nearest_neighbor_distance', 10.0)
        if nearest_neighbor_distance < 1.0:  # If any neighbor is close during emergency
            # Calculate direction away from nearest neighbor
            world = comm.snapshot()
            neighbor_positions = [np.array(s["pos"]) for peer_id, s in world.items() if peer_id != DRONE_ID]
            if neighbor_positions:
                # Find nearest neighbor
                distances = [np.linalg.norm(current_pos - p) for p in neighbor_positions]
                nearest_idx = np.argmin(distances)
                nearest_neighbor_pos = neighbor_positions[nearest_idx]
                
                # Calculate repulsion vector (away from nearest neighbor)
                repulsion_vector = current_pos - nearest_neighbor_pos
                repulsion_magnitude = 5.0  # Strong repulsion force
                if np.linalg.norm(repulsion_vector) > 0:
                    repulsion_vector = repulsion_vector / np.linalg.norm(repulsion_vector)
                    acc = repulsion_vector * repulsion_magnitude
                    print(f"[{DRONE_ID}] [EMERGENCY] Strong separation from neighbor! Distance: {nearest_neighbor_distance:.2f}m")

    # --- Reactive forward motion and obstacle avoidance fallback ---
    # Use all distance sensors to bias movement away from obstacles if no AI command is present
    front_distance = observation['distances'][0]
    back_distance = observation['distances'][1]
    left_distance = observation['distances'][2]
    right_distance = observation['distances'][3]
    reactive_yaw = 0.0
    reactive_pitch = 0.0
    reactive_roll = 0.0
    
    # Check for altitude emergency - highest priority override
    altitude_emergency = False
    if current_altitude < MIN_SAFE_ALTITUDE:
        altitude_emergency = True
        print(f"[UNIFIED] ALTITUDE EMERGENCY! Current: {current_altitude:.2f}m, Target: {MIN_SAFE_ALTITUDE}m")
    # === UNIFIED OBSTACLE AVOIDANCE SYSTEM ===
    # Single, prioritized system with consistent thresholds and responses
    # Apply only after reaching minimal safe altitude
    min_flight_altitude = 0.5
    current_altitude = altitude if 'altitude' in locals() else (gps.getValues()[2] if gps is not None else 0.0)
    
    # Initialize unified avoidance variables
    unified_pitch = 0.0
    unified_roll = 0.0
    unified_yaw = 0.0
    speed_multiplier = 1.0
    avoidance_priority = 0  # 0=none, 1=moderate, 2=close, 3=emergency
    
    if current_altitude > min_flight_altitude:
        # Get sensor distances
        front_dist = front_distance
        back_dist = back_distance
        left_dist = left_distance
        right_dist = right_distance
        
        # === PRIORITY 1: EMERGENCY AVOIDANCE (0.0 - 0.25) ===
        # Highest priority - immediate strong response
        if front_dist < 0.25:
            unified_pitch = 1.5  # Strong backward movement
            unified_yaw = 1.5    # Strong turn away
            speed_multiplier = 0.2  # Very slow
            avoidance_priority = 3
            print(f"[UNIFIED] EMERGENCY FRONT! dist={front_dist:.2f}, pitch={unified_pitch}, yaw={unified_yaw}")
            
        elif back_dist < 0.25:
            unified_pitch = -1.5  # Strong forward movement
            unified_yaw = -1.5    # Strong turn away
            speed_multiplier = 0.2  # Very slow
            avoidance_priority = 3
            print(f"[UNIFIED] EMERGENCY BACK! dist={back_dist:.2f}, pitch={unified_pitch}, yaw={unified_yaw}")
            
        elif left_dist < 0.25:
            unified_roll = -1.5   # Strong right movement
            speed_multiplier = 1.0  # Normal speed - use lateral movement only
            avoidance_priority = 3
            print(f"[UNIFIED] EMERGENCY LEFT! dist={left_dist:.2f}, roll={unified_roll}")
            
        elif right_dist < 0.25:
            unified_roll = 1.5    # Strong left movement
            speed_multiplier = 1.0  # Normal speed - use lateral movement only
            avoidance_priority = 3
            print(f"[UNIFIED] EMERGENCY RIGHT! dist={right_dist:.2f}, roll={unified_roll}")
        
        # === PRIORITY 2: CLOSE AVOIDANCE (0.25 - 0.5) ===
        # Only if no emergency triggered
        elif avoidance_priority == 0:
            if front_dist < 0.5:
                unified_pitch = 0.8   # Moderate backward movement
                unified_yaw = 0.8     # Moderate turn away
                speed_multiplier = 0.4  # Slow
                avoidance_priority = 2
                print(f"[UNIFIED] CLOSE FRONT! dist={front_dist:.2f}, pitch={unified_pitch}, yaw={unified_yaw}")
                
            elif back_dist < 0.5:
                unified_pitch = -0.8  # Moderate forward movement
                unified_yaw = -0.8    # Moderate turn away
                speed_multiplier = 0.4  # Slow
                avoidance_priority = 2
                print(f"[UNIFIED] CLOSE BACK! dist={back_dist:.2f}, pitch={unified_pitch}, yaw={unified_yaw}")
                
            elif left_dist < 0.5:
                unified_roll = -1.0   # Moderate right movement
                speed_multiplier = 1.0  # Normal speed - use lateral movement only
                avoidance_priority = 2
                print(f"[UNIFIED] CLOSE LEFT! dist={left_dist:.2f}, roll={unified_roll}")
                
            elif right_dist < 0.5:
                unified_roll = 1.0    # Moderate left movement
                speed_multiplier = 1.0  # Normal speed - use lateral movement only
                avoidance_priority = 2
                print(f"[UNIFIED] CLOSE RIGHT! dist={right_dist:.2f}, roll={unified_roll}")
        
        # === PRIORITY 3: MODERATE AVOIDANCE (0.5 - 0.7) ===
        # Only if no higher priority triggered
        elif avoidance_priority == 0:
            if front_dist < 0.7:
                unified_pitch = 0.3   # Gentle backward movement
                unified_yaw = 0.4     # Gentle turn away
                speed_multiplier = 0.6  # Moderate speed
                avoidance_priority = 1
                print(f"[UNIFIED] MODERATE FRONT! dist={front_dist:.2f}, pitch={unified_pitch}, yaw={unified_yaw}")
                
            elif back_dist < 0.7:
                unified_pitch = -0.3  # Gentle forward movement
                unified_yaw = -0.4    # Gentle turn away
                speed_multiplier = 0.6  # Moderate speed
                avoidance_priority = 1
                print(f"[UNIFIED] MODERATE BACK! dist={back_dist:.2f}, pitch={unified_pitch}, yaw={unified_yaw}")
                
            elif left_dist < 0.7:
                unified_roll = -0.6   # Gentle right movement
                speed_multiplier = 1.0  # Normal speed - use lateral movement only
                avoidance_priority = 1
                print(f"[UNIFIED] MODERATE LEFT! dist={left_dist:.2f}, roll={unified_roll}")
                
            elif right_dist < 0.7:
                unified_roll = 0.6    # Gentle left movement
                speed_multiplier = 1.0  # Normal speed - use lateral movement only
                avoidance_priority = 1
                print(f"[UNIFIED] MODERATE RIGHT! dist={right_dist:.2f}, roll={unified_roll}")
        
        # === PRIORITY 4: GENTLE AVOIDANCE (0.7 - 1.0) ===
        # Only if no higher priority triggered
        elif avoidance_priority == 0:
            # Gentle lateral steering for left/right obstacles in clear range
            if left_dist < 1.0 and left_dist >= 0.7:
                # Gentle right movement when left obstacle is detected
                unified_roll = -0.2 * (1.0 - left_dist)  # Gentle right movement (0.0 to -0.06)
                print(f"[UNIFIED] GENTLE LEFT! dist={left_dist:.2f}, roll={unified_roll:.3f}")
            elif right_dist < 1.0 and right_dist >= 0.7:
                # Gentle left movement when right obstacle is detected
                unified_roll = 0.2 * (1.0 - right_dist)  # Gentle left movement (0.0 to 0.06)
                print(f"[UNIFIED] GENTLE RIGHT! dist={right_dist:.2f}, roll={unified_roll:.3f}")
            
            # Full speed for gentle avoidance
            speed_multiplier = SPEED_MULTIPLIER  # Full speed
            print(f"[UNIFIED] GENTLE AVOIDANCE! Using full speed {SPEED_MULTIPLIER}x")
        
        # === NORMAL OPERATION (truly clear path) ===
        if avoidance_priority == 0 and unified_roll == 0.0:
            speed_multiplier = SPEED_MULTIPLIER  # Full speed
            print(f"[UNIFIED] CLEAR PATH! Using full speed {SPEED_MULTIPLIER}x")
    
    else:
        # Before takeoff - no avoidance
        speed_multiplier = 0.0
        print(f"[UNIFIED] PRE-TAKEOFF! No avoidance, altitude={current_altitude:.2f}m")


    # --- UNIFIED CONTROL INTEGRATION ---
    # Combine leader-follower control with unified obstacle avoidance
    front_dist, back_dist, left_dist, right_dist = observation['distances']
    
    # Base control from leader-follower system with speed adjustment
    # Apply speed multiplier from unified avoidance system
    control_multiplier = speed_multiplier
    
    # Primary control: goal-seeking and formation (scaled by avoidance priority)
    pitch_disturbance = -thrust_adjustment * control_multiplier
    roll_disturbance = 0.0  # Roll handled by formation positioning
    yaw_disturbance = yaw_rate * control_multiplier
    
    # Priority control system with BLENDED approach (not override):
    if altitude_emergency:
        # Priority 1: Altitude emergency - force upward movement
        pitch_disturbance = -2.0  # Strong upward movement
        roll_disturbance = 0.0    # No lateral movement when too low
        yaw_disturbance = 0.0     # No turning when too low
        print(f"[UNIFIED] ALTITUDE OVERRIDE! Forcing upward movement")
    else:
        # Priority 2 & 3: BLEND goal-seeking with obstacle avoidance
        # Base goal-seeking control (always active)
        base_pitch = -thrust_adjustment * control_multiplier
        base_roll = 0.0  # Roll handled by formation positioning
        base_yaw = yaw_rate * control_multiplier
        
        # Add boids contribution
        base_pitch += float(acc[0]) * 0.05  # Forward/backward from boids
        base_roll += float(acc[1]) * 0.05   # Left/right from boids
        base_yaw += float(acc[2]) * 0.1     # Yaw from boids
        
        # Blend with obstacle avoidance based on priority
        if avoidance_priority > 0:
            # Blend avoidance with goal-seeking (avoidance gets more weight for higher priority)
            avoidance_weight = avoidance_priority / 3.0  # 0.33, 0.67, 1.0 for priorities 1,2,3
            goal_weight = 1.0 - avoidance_weight
            
            pitch_disturbance = (base_pitch * goal_weight) + (unified_pitch * avoidance_weight)
            roll_disturbance = (base_roll * goal_weight) + (unified_roll * avoidance_weight)
            yaw_disturbance = (base_yaw * goal_weight) + (unified_yaw * avoidance_weight)
            
            print(f"[UNIFIED] BLENDED! Priority={avoidance_priority}, goal_weight={goal_weight:.2f}, avoid_weight={avoidance_weight:.2f}")
            print(f"[UNIFIED] Final: pitch={pitch_disturbance:.2f}, roll={roll_disturbance:.2f}, yaw={yaw_disturbance:.2f}")
        else:
            # Normal operation - pure goal-seeking + boids
            pitch_disturbance = base_pitch
            roll_disturbance = base_roll
            yaw_disturbance = base_yaw
            print(f"[UNIFIED] NORMAL! Pure goal-seeking + boids, pitch={pitch_disturbance:.2f}, roll={roll_disturbance:.2f}, yaw={yaw_disturbance:.2f}")

    # Apply control only after all drones have taken off
    if not all_taken_off:
        # Before takeoff, use minimal control - but don't completely stop
        # Allow some movement to help with takeoff coordination
        if current_altitude < min_flight_altitude:
            # This drone needs to take off
            pitch_disturbance = -1.0  # Upward movement
            roll_disturbance = 0.0
            yaw_disturbance = 0.0
            print(f"[UNIFIED] TAKEOFF! This drone needs to take off, altitude={current_altitude:.2f}m")
        else:
            # This drone is ready, but others aren't - use minimal movement
            pitch_disturbance *= 0.3  # Reduce but don't stop
            roll_disturbance *= 0.3
            yaw_disturbance *= 0.3
            print(f"[UNIFIED] WAITING! This drone ready, others not, using reduced control")

    # --- PID-like stabilization ---
    roll_input = K_ROLL_P * max(-1.0, min(1.0, roll)) + roll_velocity + roll_disturbance
    pitch_input = K_PITCH_P * max(-1.0, min(1.0, pitch)) + pitch_velocity + pitch_disturbance
    clamped_diff_alt = max(-1.0, min(1.0, target_altitude - altitude + K_VERTICAL_OFFSET))
    vertical_input = K_VERTICAL_P * (clamped_diff_alt ** 3.0)

    fl = K_VERTICAL_THRUST + vertical_input - roll_input + pitch_input - yaw_disturbance
    fr = K_VERTICAL_THRUST + vertical_input + roll_input + pitch_input + yaw_disturbance
    rl = K_VERTICAL_THRUST + vertical_input - roll_input - pitch_input + yaw_disturbance
    rr = K_VERTICAL_THRUST + vertical_input + roll_input - pitch_input - yaw_disturbance

    # --- Safety clamp: prevent NaN or infinite values ---
    motor_speeds = [fl, fr, rl, rr]
    motor_speeds = [0.0 if not np.isfinite(v) else v for v in motor_speeds]

    front_left_motor.setVelocity(motor_speeds[0])
    front_right_motor.setVelocity(-motor_speeds[1])
    rear_left_motor.setVelocity(-motor_speeds[2])
    rear_right_motor.setVelocity(motor_speeds[3])
    # Debug: Print actual motor speeds sent to each motor
    print(f"[{DRONE_ID}] [DEBUG] Motor speeds: FL={motor_speeds[0]:.2f}, FR={-motor_speeds[1]:.2f}, RL={-motor_speeds[2]:.2f}, RR={motor_speeds[3]:.2f}")

    # --- Debug prints ---
    if robot.getTime() % 1.0 < 0.1:  # Print every ~1 second
        fd = observation['distances'][0]
        fs = observation.get('front_source', 'n/a')
        print(f"[{DRONE_ID}] Action: {current_action}, Thrust: {thrust_adjustment:.2f}, Yaw: {yaw_rate:.2f}, front={fd:.2f} ({fs})")

    # On close approach, emit an explicit debug line once per second
    if front_distance < 0.5 and robot.getTime() % 1.0 < 0.1:
        print(f"[{DRONE_ID}] CLOSE OBSTACLE: front={front_distance:.2f}, src={observation.get('front_source','n/a')}, yaw_add={reactive_yaw:.2f}")
