import json
from controller import Supervisor
from agents.fuzzy_controller import create_fuzzy_controller
from agents import boids
from agents import config as C
import numpy as np

# === Initialization ===
TIME_STEP = 64
supervisor = Supervisor()

# Get drone references from the Webots world
NUM_DRONES = 10
all_drones = [supervisor.getFromDef(f"drone{i}") for i in range(NUM_DRONES)]
drones = [d for d in all_drones if d is not None]
if len(drones) < NUM_DRONES:
    print(f"Warning: Only {len(drones)} drones found. Check your world file for missing entries.")

# Initialize drones in a star (pentagram) shape
def double_arrowhead_positions(center, spacing, num_points):
    positions = []
    half = num_points // 2
    for i in range(half):
        x = center[0] - spacing * i
        y = center[1]
        z = center[2] + spacing * i
        positions.append(np.array([x, y, z]))
    for i in range(half, num_points):
        x = center[0] - spacing * (num_points - i - 1)
        y = center[1]
        z = center[2] - spacing * (num_points - i - 1)
        positions.append(np.array([x, y, z]))
    return positions

center = [0, 1, 0]  # Adjust Y as needed for your world
spacing = 1.5
initial_positions = double_arrowhead_positions(center, spacing, NUM_DRONES)

# State variable to track initial drone positions
drone_initial_states = {f"drone{i}": {"pos": initial_positions[i].tolist(), "vel": [0, 0, 0]} for i in range(NUM_DRONES)}

fuzzy_sim = create_fuzzy_controller()

# === Main Loop ===
while supervisor.step(TIME_STEP) != -1:
    current_positions = {}
    current_velocities = {}
    
    # On first step, set drones to star positions and communicate to agents
    if supervisor.getTime() < TIME_STEP / 1000.0:
        for i, drone in enumerate(drones):
            if drone:
                # Set initial position and broadcast it to the drone agent
                drone.getField("translation").setSFVec3f(initial_positions[i].tolist())
                current_positions[f"drone{i}"] = initial_positions[i]
                current_velocities[f"drone{i}"] = np.zeros(3)
                
    else:
        # Get current positions and velocities from each drone
        for i, drone in enumerate(drones):
            if drone:
                pos = np.array(drone.getField("translation").getSFVec3f())
                # For a true velocity, you'd need to store the previous position and time,
                # but for this simplified model, we'll assume the velocity is part of the state
                # that gets updated by the drone's own controller or simulated here.
                # A better approach would be to receive this from the agents.
                current_positions[f"drone{i}"] = pos
                current_velocities[f"drone{i}"] = np.zeros(3) # Simplified for example
    
    # === Boids Logic ===
    # Here, you would implement the logic to gather drone data (pos, vel)
    # and compute the new Boids-based target for each one.
    
    # This is a placeholder for the Boids computation
    new_targets = {}
    for i in range(NUM_DRONES):
        drone_name = f"drone{i}"
        
        # This part of the code is conceptual. A proper implementation would
        # involve a communication channel to get real-time pos/vel from agents.
        # For now, we use the supervisor's knowledge of the world.
        if drone_name in current_positions:
            pos = current_positions[drone_name]
            vel = current_velocities[drone_name]
            
            # Simplified boids step where the goal is the initial formation position
            acc = boids.step(pos, vel, {}, goal=initial_positions[i])
            
            # Calculate new target position based on Boids acceleration
            # This is a simplified integration. For a robust system, you'd
            # integrate the acceleration to get a new velocity and position.
            new_target = pos + acc * TIME_STEP / 1000.0
            
            new_targets[drone_name] = new_target.tolist()

    # The supervisor now simply broadcasts the target positions
    # to the individual drone controllers.
    # Note: A real implementation requires a way to send this data.
    # We will assume a mechanism (like a UDP broadcast) exists.

    # This part is crucial:
    # The supervisor should send commands to the drones, not move them directly.
    # Example:
    # supervisor.broadcast_targets(new_targets)

    # For the purpose of this supervisor-only fix, we will directly set positions
    # based on the boids calculation, but the best practice is to send commands.
    for i, drone in enumerate(drones):
        if drone and f"drone{i}" in new_targets:
            drone.getField("translation").setSFVec3f(new_targets[f"drone{i}"])