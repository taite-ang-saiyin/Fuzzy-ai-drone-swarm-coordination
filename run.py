
import argparse
import os
import numpy as np
import time
import sys
from eval.logger import Logger
from agents.config import get_config
from agents.fuzzy_controller import create_fuzzy_controller
from agents import boids
from agents.goals import WaypointManager
from agents.comm import UdpPeer

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--n', type=int, default=5)
	parser.add_argument('--scenario', type=str, default='open_field', choices=['open_field','single_obstacle'])
	parser.add_argument('--log', type=str, default='eval/logs')
	parser.add_argument('--duration', type=int, default=30, help='Simulation duration in seconds')
	args = parser.parse_args()

	os.makedirs(args.log, exist_ok=True)
	loggers = [Logger(args.log, i) for i in range(args.n)]
	print(f"[Run] Starting scenario '{args.scenario}' with {args.n} drones. Logging to {args.log}")

	# Get configuration
	C = get_config()
	
	# Initialize simulation parameters
	WORLD_SIZE = 50.0
	SENSOR_RANGE = 8.0
	DRONE_RADIUS = 0.4
	OBSTACLE_RADIUS = 1.5
	DT = 0.1  # 100ms timestep
	
	# Initialize drones with proper spacing
	drone_positions = []
	drone_velocities = []
	drone_orientations = []
	
	# Create initial formation (circular pattern)
	init_radius = 8.0
	init_altitude = 2.0
	for i in range(args.n):
		angle = 2 * np.pi * i / args.n
		x = init_radius * np.cos(angle)
		y = init_radius * np.sin(angle)
		z = init_altitude
		drone_positions.append(np.array([x, y, z]))
		drone_velocities.append(np.zeros(3))
		drone_orientations.append(0.0)
	
	# Initialize obstacles based on scenario
	obstacles = []
	if args.scenario == 'single_obstacle':
		obstacles.append(np.array([WORLD_SIZE/2, WORLD_SIZE/2, 1.5]))
	elif args.scenario == 'open_field':
		# Add some random obstacles
		for _ in range(3):
			obs_pos = np.array([
				np.random.uniform(5, WORLD_SIZE-5),
				np.random.uniform(5, WORLD_SIZE-5),
				1.5
			])
			obstacles.append(obs_pos)
	
	# Initialize controllers
	fuzzy_controllers = [create_fuzzy_controller() for _ in range(args.n)]
	
	# Initialize waypoint manager with shared goals
	waypoint_manager = WaypointManager(global_waypoints=[
		np.array([WORLD_SIZE-5, WORLD_SIZE-5, 2.0]),
		np.array([5, WORLD_SIZE-5, 2.0]),
		np.array([5, 5, 2.0]),
		np.array([WORLD_SIZE-5, 5, 2.0])
	])
	
	# Shared goal for all drones (leader-follower system)
	shared_goal = waypoint_manager.get_global_goal()
	goal_reached_threshold = 3.0
	
	# Initialize communication (simulated)
	comm_peers = []
	for i in range(args.n):
		peer_ports = [C['comms']['base_port'] + j for j in range(args.n) if j != i]
		peer = UdpPeer(f"drone{i}", C['comms']['base_port'] + i, peer_ports)
		peer.start()
		comm_peers.append(peer)
	
	# Simulation loop
	start_time = time.time()
	step = 0
	
	print(f"[Run] Starting simulation for {args.duration} seconds...")
	
	while time.time() - start_time < args.duration:
		step_start = time.time()
		
		# Update each drone
		for i in range(args.n):
			pos = drone_positions[i]
			vel = drone_velocities[i]
			orientation = drone_orientations[i]
			
			# Get sensor readings (simplified)
			sensor_distances = []
			for direction in ['front', 'back', 'left', 'right']:
				# Calculate sensor direction based on orientation
				if direction == 'front':
					dir_vec = np.array([np.cos(orientation), np.sin(orientation), 0])
				elif direction == 'back':
					dir_vec = np.array([-np.cos(orientation), -np.sin(orientation), 0])
				elif direction == 'left':
					dir_vec = np.array([-np.sin(orientation), np.cos(orientation), 0])
				else:  # right
					dir_vec = np.array([np.sin(orientation), -np.cos(orientation), 0])
				
				# Find distance to nearest obstacle
				min_dist = SENSOR_RANGE
				for obs in obstacles:
					rel = obs - pos
					proj = np.dot(rel, dir_vec)
					if 0 < proj < SENSOR_RANGE:
						closest = pos + proj * dir_vec
						dist_to_obs = np.linalg.norm(obs - closest)
						if dist_to_obs < OBSTACLE_RADIUS + DRONE_RADIUS:
							min_dist = min(min_dist, proj)
				
				sensor_distances.append(min_dist / SENSOR_RANGE)  # Normalize
			
			# Get neighbor information
			neighbors = []
			neighbor_distances = []
			for j in range(args.n):
				if i != j:
					dist = np.linalg.norm(pos - drone_positions[j])
					if dist < C['boids']['perception_radius']:
						neighbors.append((j, drone_positions[j], drone_velocities[j], dist))
						neighbor_distances.append(dist)
			
			nearest_neighbor_distance = min(neighbor_distances) if neighbor_distances else C['boids']['perception_radius']
			nearest_neighbor_distance = min(nearest_neighbor_distance / C['boids']['perception_radius'], 1.0)  # Normalize
			
			# All drones use the same shared goal
			goal = shared_goal
			
			# Apply fuzzy controller for obstacle avoidance
			fuzzy_output = fuzzy_controllers[i].get_advanced_avoidance_policy(
				sensor_distances, 
				nearest_neighbor_distance, 
				f"drone{i}"
			)
			
			# Apply boids algorithm for formation with shared goal
			world_states = {}
			for j in range(args.n):
				world_states[f"drone{j}"] = {
					'pos': drone_positions[j],
					'vel': drone_velocities[j]
				}
			
			boids_acc = boids.step(pos, vel, world_states, goal=goal)
			
			# Combine fuzzy avoidance with boids formation
			thrust_adj = fuzzy_output.get('thrust_adjustment', 0.0)
			yaw_rate = fuzzy_output.get('yaw_rate', 0.0)
			
			# Calculate movement toward shared goal
			goal_vector = goal - pos
			goal_distance = np.linalg.norm(goal_vector)
			
			# Determine if this is the leader (drone 0) or follower
			is_leader = (i == 0)
			
			if is_leader:
				# Leader: move directly toward goal with formation consideration
				goal_direction = goal_vector / (goal_distance + 1e-6)
				
				# Blend goal-seeking with formation (boids)
				goal_weight = 0.8
				formation_weight = 0.2
				
				if np.linalg.norm(boids_acc) > 0:
					formation_direction = boids_acc / np.linalg.norm(boids_acc)
				else:
					formation_direction = goal_direction
				
				# Calculate speed based on distance to goal
				speed = min(3.0, max(0.5, goal_distance * 0.3))
				speed += thrust_adj * 0.5  # Adjust for obstacle avoidance
				speed = max(0.1, min(5.0, speed))
				
				# Final velocity for leader
				new_vel = goal_weight * goal_direction * speed + formation_weight * formation_direction * speed
				
			else:
				# Follower: maintain formation relative to leader
				leader_pos = drone_positions[0]
				leader_vel = drone_velocities[0]
				
				# Calculate desired formation position (V-formation)
				formation_spacing = 4.0
				formation_angle = np.radians(30)
				
				# Get leader's heading
				if np.linalg.norm(leader_vel) > 0.1:
					leader_heading = np.arctan2(leader_vel[1], leader_vel[0])
				else:
					leader_heading = 0.0
				
				# Calculate formation offset
				side = -1 if i % 2 == 1 else 1  # Alternate sides
				rank = (i + 1) // 2
				
				# Formation position relative to leader
				dx = -rank * formation_spacing * np.cos(formation_angle)
				dy = side * rank * formation_spacing * np.sin(formation_angle)
				
				# Rotate offset by leader's heading
				cos_h = np.cos(leader_heading)
				sin_h = np.sin(leader_heading)
				offset_x = dx * cos_h - dy * sin_h
				offset_y = dx * sin_h + dy * cos_h
				
				desired_pos = leader_pos + np.array([offset_x, offset_y, 0])
				formation_vector = desired_pos - pos
				
				# Blend formation with goal-seeking and avoidance
				formation_weight = 0.6
				goal_weight = 0.3
				avoidance_weight = 0.1
				
				# Normalize directions
				if np.linalg.norm(formation_vector) > 0:
					formation_direction = formation_vector / np.linalg.norm(formation_vector)
				else:
					formation_direction = np.array([1.0, 0.0, 0.0])
				
				goal_direction = goal_vector / (goal_distance + 1e-6)
				
				# Calculate speed
				speed = min(2.5, max(0.3, np.linalg.norm(formation_vector) * 0.4))
				speed += thrust_adj * 0.3
				speed = max(0.1, min(4.0, speed))
				
				# Final velocity for follower
				new_vel = (formation_weight * formation_direction + 
						  goal_weight * goal_direction + 
						  avoidance_weight * np.array([np.cos(orientation + yaw_rate * DT), 
													   np.sin(orientation + yaw_rate * DT), 0])) * speed
			
			# Update orientation based on movement direction
			if np.linalg.norm(new_vel) > 0.1:
				new_orientation = np.arctan2(new_vel[1], new_vel[0])
			else:
				new_orientation = orientation
			
			# Update position and velocity
			drone_positions[i] = pos + new_vel * DT
			drone_velocities[i] = new_vel
			drone_orientations[i] = new_orientation
			
			# Broadcast state
			comm_peers[i].broadcast(drone_positions[i], drone_velocities[i])
			
			# Log data
			goal_dx, goal_dy, goal_dz = goal - pos
			row = [
				time.time(), step, i,
				pos[0], pos[1], pos[2],  # position
				vel[0], vel[1], vel[2],  # velocity
				orientation,  # yaw
				min(sensor_distances),  # d_obstacle (closest)
				len(neighbors),  # neighbor_count
				np.mean(neighbor_distances) if neighbor_distances else 0.0,  # neighbor_avg_dist
				thrust_adj, yaw_rate,  # control outputs
				0,  # collided (simplified)
				goal_dx, goal_dy, goal_dz,  # goal vector
				0.0  # reward (simplified)
			]
			loggers[i].log(row)
		
		# Check if global goal reached (only leader needs to reach it)
		leader_distance = np.linalg.norm(drone_positions[0] - shared_goal)
		if leader_distance < goal_reached_threshold:
			# Move to next waypoint
			waypoint_manager.global_goal_reached(drone_positions[0], thresh=goal_reached_threshold)
			shared_goal = waypoint_manager.get_global_goal()
			print(f"[Run] Goal reached! Moving to next waypoint: {shared_goal}")
		
		step += 1
		
		# Maintain timing
		elapsed = time.time() - step_start
		sleep_time = DT - elapsed
		if sleep_time > 0:
			time.sleep(sleep_time)
	
	# Cleanup
	for peer in comm_peers:
		peer.stop()
	for l in loggers:
		l.close()
	
	print(f"[Run] Simulation completed. {step} steps executed.")

if __name__ == '__main__':
	main()
